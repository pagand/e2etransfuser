import json
import os
import sys
import logging
import asyncio
import time
from tqdm import tqdm

Folder =[]

# function that takes a queue, minmum length (m) and json file address as input and write 'vel' element 
# of json file into queue and pop the last element of queue
def update_queue(queue_s, queue_t, queue_b, json_file):
    # read json file
    try:
        with open(json_file) as f:
            data = json.load(f)
    except:
        logging.error('Failed to read {} file'.format(json_file))
        return None
    # append tuple element of json file into queue
    queue_s.append(data['steer'] if type(data['steer'])==type(0.1) else data['steer'][0])
    queue_t.append(data['throttle'] if type(data['throttle'])==type(0.1) else data['throttle'][0])
    queue_b.append(data['brake'] if type(data['brake'])==type(True) else data['brake'][0])

    # pop the last element of queue
    queue_s.pop(0)
    queue_t.pop(0)
    queue_b.pop(0)

    return queue_s, queue_t, queue_b

# function that takes a queue and json file address as input and write queue into 'vel' element of json 
def update_json(queue_s, queue_t, queue_b, json_file):
    #read a json file
    
    with open(json_file) as f:
        data = json.load(f)
    
    
    # write queue into control elements of json
    data['steer'],data['throttle'], data['brake']  = queue_s, queue_t, queue_b 

    # dump data to json file
    try:
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except:
        return False

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

# function that takes folder address as input and call update_queue function for each json file in the folder sorted according to their names
# then wait to have at least m elements in queue and call update_json function for each json file in the folder sorted according to their names
@background
def get_queue(folder,seq_len):
    start = time.time()
    
    # get list of json files in the folder sorted according to their names 
    files = sorted(os.listdir(folder))

    # initialize queue
    queue_s = [0 for _ in range(seq_len)]
    queue_t = [0 for _ in range(seq_len)]
    queue_b = [0 for _ in range(seq_len)]

    # call update_queue function for each json file in the folder sorted according to their names
    for ind, file in enumerate(files):
        queue_s, queue_t, queue_b = update_queue(queue_s, queue_t, queue_b, os.path.join(folder, file))
        if not queue_s or not queue_b or not queue_t:
            return None
        if ind>=seq_len-1:
            log = update_json(queue_s, queue_t, queue_b, os.path.join(folder, files[ind-seq_len+1]))
            # if update_json return False, write the failed file name in logging.error
            if not log:
                if log == False:
                    logging.error('failed to write {} file in {} folder'.format(files[ind-seq_len+1], folder))
                
                
    # continue to update json for the last m files 
    for i in range(seq_len-1,0, -1):
        queue_s.append(queue_s[-1])
        queue_t.append(queue_t[-1])
        queue_b.append(queue_b[-1])
        queue_s.pop(0)
        queue_t.pop(0)
        queue_b.pop(0)
        update_json(queue_s,queue_t, queue_b, os.path.join(folder, files[-i]))
    
    #logging.info('Processing folder: {} Executed elpsed time: {}'.format(folder, time.time()-start))
    Folder.append(folder.split('/')[-2])
    # return folder.split('/')[-1] 



def main(root,seq_len,nt):
    sub_folders = os.listdir(root)
    ins = []
    prog_bar = tqdm(total=nt)
    for folders in sub_folders:
        rfolders = os.path.join(root, folders)
        sub_routes = os.listdir(rfolders)
        for routes in sub_routes:
            rroutes = os.path.join(rfolders, routes)
            samples = os.listdir(rroutes)
            for sample in samples:
                rsample = os.path.join(rroutes, sample)
                if os.path.isdir(rsample):
                    # to keep track
                    ins.append(sample)
                    get_queue(os.path.join(rsample, 'measurements'),seq_len)
                    #folder = get_queue(os.path.join(rsample, 'measurements'),seq_len)
                    #outs.append(folder)
                    #prog_bar.set_postfix(sample)
                    prog_bar.update(1)
            time.sleep(20*len(samples)/7)
    prog_bar.close()     
    return ins#list(set(ins)-set(outs)



# main function that call get_stack for a sample data folder
if __name__ == '__main__':
    logging.basicConfig(filename='./utilx/log.txt', level=logging.DEBUG)
    start = time.time()
    data_address = '/localhome/pagand/projects/e2etransfuser/transfuser_pmlr/data0'
    ntotal = 45 # total folders
    ins = main(data_address,seq_len=3,nt = ntotal)
    print('Waiting for all background process to finish ...')
    time.sleep(ntotal/10)
    logging.info('Number of incomplete files: {}'.format(len(ins)- len(Folder)))
    logging.info('Incompleted folders: {}'.format(list(set(ins)-set(Folder))))
    # all_tasks = asyncio.all_tasks()
    # await asyncio.wait(all_tasks)

    logging.info('TOTAL elapsed time {}'.format( time.time()-start))