import json
from traceback import format_exc

import flask_sock
import hivemind
from flask import request as http_request

import config
from app import models, sock
from utils import safe_decode
import multiprocessing as mp

import threading

from multiprocessing import Manager
from transformers import TextIteratorStreamer

logger = hivemind.get_logger(__file__)

#manager = Manager()       
INFERENCE_PATH = {} #manager.dict()
GLOBAL_MAP = {}
lock = threading.Lock()
isDummyRunning = False

def run_dummy_session(model,tokenizer):
    global GLOBAL_MAP
    with lock:
        if len(GLOBAL_MAP) > 0:
            logger.info(f"HIVE: DummyPath via: {GLOBAL_MAP}")
            return
    with model.inference_session(max_length=25) as session:
        found = False
        while True:
            with lock:
                if len(GLOBAL_MAP) > 0 or found == True:
                    logger.info(f"HIVE: DummyPath via: {GLOBAL_MAP}")
                    return

            inputs = "hi"
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            #n_input_tokens = inputs.shape[1]
            _ = model.generate(inputs=inputs,do_sample=False,max_new_tokens=1,session=session)
            sessionlist = session._server_sessions
            for sid in sessionlist:
                found = True
                block_range = str(sid.span.start) + ":" + str(sid.span.end)
                ip_addr = str(sid.span.server_info.public_name)
                peer_id = str(sid.span.peer_id)
                with lock:
                    GLOBAL_MAP[block_range] = ip_addr + " (..." + peer_id[-5:] +")"
        
        
         


@sock.route("/api/v2/generate")
def ws_api_generate(ws):
    
    try:
        request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
        assert request["type"] == "open_inference_session"
        model_name = request["model"]
        max_length = request["max_length"]
        logger.info(f"ws.generate.open(), {model_name=}, {max_length=}, {http_request.origin=}")

        model, tokenizer, backend_config = models[model_name]
        if not backend_config.public_api and http_request.origin != f"{http_request.scheme}://{http_request.host}":
            raise ValueError(f"We do not provide public API for {model_name} due to license restrictions")

        #with model.inference_session(max_length=max_length) as session:
        
        ws.send(json.dumps({"ok": True}))
        
        global isDummyRunning
        if not isDummyRunning:
            #mp.set_start_method('spawn')
            dummySession = threading.Thread(target=run_dummy_session, args=(model,tokenizer))
            dummySession.start()
            isDummyRunning = True
        
        request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
        assert request["type"] == "generate"
        inputs = request.get("inputs")

        logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
        n_input_tokens = 0
        nm_tokens = 0
        if inputs is not None:
            
            temp0 = repr(inputs).split("###Human:")
            temp1 = ""
            UserInput = ""
            if len(temp0)> 0:
                temp1 = temp0[len(temp0)-1].split("###")
                if len(temp1) > 0:
                    UserInput = temp1[0].strip()
                    UserInput = UserInput.replace('Human:', '')
            logger.info(f"ws.generate.step(), RawInputs={repr(inputs)}")
            #inputs = "Human: " + UserInput + "\n\nAssistant:"
            inputs = UserInput
            nm_tokens = len(inputs.split())
            
                    
            logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            n_input_tokens = inputs.shape[1]
        else:
            n_input_tokens = 0
        
        streamer = TextIteratorStreamer(tokenizer)
        thread = threading.Thread(target=model.generate,kwargs=dict(inputs=inputs,
                        do_sample=request.get("do_sample", False),
                        temperature=request.get("temperature"),
                        top_k=request.get("top_k"),
                        top_p=request.get("top_p"),
                        repetition_penalty=request.get("repetition_penalty"),
                        max_length=request.get("max_length"),
                        max_new_tokens=100,
                        streamer=streamer))
        thread.start()
        max_token = 100
        index = 0 
        stop = False
        for outputs in streamer:
            global GLOBAL_MAP
            token_count = 0
            route_json = {}
            with lock:
                route_json = json.dumps(GLOBAL_MAP)
            #HIVE END
            token_count = len(outputs.split())
            stop_sequence = request.get("stop_sequence")
            if ((outputs.endswith(stop_sequence)) or (outputs.endswith("\n-----\n")) or (index >= max_token)):
                stop = True
                #outputs = ""
                #token_count = 0 
            if ((outputs.endswith("-----")) or (outputs.find("-----")!=-1)):
                stop = True
                #outputs = ""
                #token_count = 0
            if index >= max_token:
                stop = True
                #outputs = ""
                #token_count = 0
            if index > nm_tokens-1: #(n_input_tokens-5):
                ws.send(json.dumps({"ok": True, "outputs": outputs, "stop": stop, "token_count": token_count, "route":route_json}))
            incr = len(outputs.split())
            index+=incr
            logger.info(f"HIVE Incr Ouptput = {outputs}")

            if stop:
                break
            #outputs = [text]
            '''   
            stop_sequence = request.get("stop_sequence")
            extra_stop_sequences = request.get("extra_stop_sequences")
            if extra_stop_sequences is not None:
                cont_token = tokenizer(stop_sequence, return_tensors="pt")["input_ids"].to(config.DEVICE)
                if cont_token.shape != (1, 1):
                    raise ValueError("extra_stop_sequences require stop_sequence length to be exactly 1 token")

            all_outputs = ""
            delta_q = []
            stop = False
            while not stop:
                
                #delta = outputs[0, n_input_tokens:].tolist()
                #outputs = safe_decode(tokenizer, delta_q + delta)
                inputs = None  # Inputs are passed only for the 1st token of the bot's response
                n_input_tokens = 0
                combined = all_outputs + outputs
                stop = stop_sequence is None or (
                    "falcon-180B" not in model_name and combined.endswith(stop_sequence)
                )
                if extra_stop_sequences is not None:
                    for seq in extra_stop_sequences:
                        if combined.endswith(seq) or combined.endswith("\\n\\n"):
                            stop = True
                            #session.last_token_id = cont_token
                if not stop and outputs[-10:].find("\ufffd") > -1:
                    # If there's a replacement character, keep getting more tokens
                    # until we can decode properly
                    #delta_q = delta_q + delta
                    logger.info(f"ws.generate.append_retry(), all_outputs={repr(combined)}")
                else:
                    all_outputs = combined
                    token_count = len(outputs.split())#len(delta_q + delta)
                    delta_q = []
                    logger.info(f"ws.generate.step(), all_outputs={repr(all_outputs)}, stop={stop}")
                    #HIVE START
                    
                    sessionlist = session._server_sessions
                    route_map = {}
                    for sid in sessionlist:
                    block_range = str(sid.span.start) + ":" + str(sid.span.end)
                    ip_addr = str(sid.span.server_info.public_name)
                    peer_id = str(sid.span.peer_id)
                    route_map[block_range] = ip_addr + " (..." + peer_id[-5:] +")"
                    
                    #logger.info(f"HIVE: PeerID = {sid.span.peer_id}; BLOCKS = {sid.span.start},{sid.span.end}")
                    global GLOBAL_MAP
                    route_json = {}
                    with lock:
                        route_json = json.dumps(GLOBAL_MAP)
                    #HIVE END
                    ws.send(json.dumps({"ok": True, "outputs": outputs, "stop": stop, "token_count": token_count, "route":route_json}))
                    '''
    except flask_sock.ConnectionClosed:
        pass
    except Exception:
        logger.warning("ws.generate failed:", exc_info=True)
        ws.send(json.dumps({"ok": False, "traceback": format_exc()}))
    finally:
        logger.info(f"ws.generate.close()")
