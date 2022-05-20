from datetime import datetime
from typing import List, Dict, Union, Tuple, Callable
from itertools import product
from math import log
import json
from random import choice, sample
import matplotlib.pyplot as plt
import os
from pprint import pprint

def sum_product(p):
    n=len(p[0])
    return [[sum([pi[y][x] for pi in p]) for x in range(n)] for y in range(n)]

def is_valid(board:List[List[int]])->bool:
    return max(max(board,key=lambda x: max(x)))==1

def calculate_positions(ships:List[List[List[int]]], size:int):
    board=[[0 for j in range(size)] for i in range(size)]
    ship_boards=[[] for _ in ships]
    cb=[[0 for j in range(size)] for i in range(size)]#board.copy()
    for si,tship in enumerate(ships): 
        for ship in tship:
            for y in range(size-ship[1]+1):
                for x in range(size-ship[0]+1):
                    for i in range(ship[0]):
                        for j in range(ship[1]):
                            cb[y+j][x+i]=1
                    ship_boards[si].append(cb)
                    cb=[[0 for j in range(size)] for i in range(size)]
    possible_boards=[]
    for pboard in product(*ship_boards):
        board=sum_product(pboard)
        if is_valid(board): possible_boards.append(board)
    return possible_boards

def generate_board(ships:List[List[List[int]]], size:int, choices:List[int], positions:List[List[int]])->List[List[int]]:
    board=[[0 for j in range(size)] for i in range(size)]
    for i,tship in enumerate(ships):
        ship=tship[choices[i]]
        for i in range(ship[0]):
            for j in range(ship[1]):
                board[positions[i][1]+j][positions[i][0]+i]=1
    return board

def print_board(board:Union[List[List[int]],List[List[float]]]):
    print(80*"-")
    for row in board:
        print([float("{:.3f}".format(v)) for v in row])
    print(80*"-")

def save_data(ships:List[List[List[int]]], size:int):
    pbs:List[List[List[int]]]=calculate_positions(ships, size)
    data={"ships":ships,"size":size}
    with open("boards.json", "w+") as f:
        json.dump(pbs,f,indent=3)
    with open("meta.json", "w+") as f:
        json.dump(data,f,indent=3)

def load_data()->Tuple[List[List[List[int]]], Dict[str, Union[int, List[List[List[int]]]]]]:
    with open("boards.json", "r+") as f:
        boards=json.load(f)
    with open("meta.json", "r+") as f:
        data=json.load(f)
    return boards,data

def get_goal(boards:List[List[List[int]]])->List[List[int]]:
    return choice(boards)

def get_state(board:List[List[int]],x:int,y:int)->int:
    return board[y][x]

def filter_after_shot(boards:List[List[List[int]]],board:List[List[int]],x:int,y:int)->Tuple[List[List[List[int]]],List[List[List[int]]]]:
    d_state=get_state(board,x,y)
    valids=[]
    invalids=[]
    for b in boards:
        if b[y][x]==d_state:
            valids.append(b)
        else:
            invalids.append(b)
    return invalids, valids 

def calculate_entropy(boards:List[List[List[int]]],x:int,y:int)->float:
    n=len(boards)
    invalids=[]
    valids=[]
    for b in boards:
        if b[y][x]==1:
            valids.append(b)
        else:
            invalids.append(b)
    ni=len(invalids)
    nv=len(valids)
    if ni==n or nv==n: return 0
    else: return -(ni/n * log(ni/n,2) + nv/n * log(nv/n,2))

def calculate_entropy_map(boards:List[List[List[int]]], size:int)->List[List[float]]:
    return [[calculate_entropy(boards,x,y) for x in range(size)] for y in range(size)]

def calculate_expectation_map(boards:List[List[List[int]]], size:int)->List[List[float]]:
    n=len(boards)
    avg=[[0 for x in range(size)] for y in range(size)]
    for b in boards:
        for x in range(size):
            for y in range(size):
               avg[y][x] +=  b[y][x]
    return [[avg[y][x]/n for x in range(size)] for y in range(size)]
    
def draw_state(boards:List[List[List[int]]],goal:List[List[int]],shots:List[List[int]],entro_map:List[List[float]],size:int,fname:str):

    plt.clf()
    exp_entro_map=calculate_entropy_map(sample(boards,min(1000,len(boards))),size)
    exp_goal=calculate_expectation_map(boards,size)
    shots=[[[x,y] in shots for x in range(size)] for y in range(size)]
    hits=[[(shots[y][x]==1) and (goal[y][x]==1) for x in range(size)] for y in range(size)]
    diff_goal=[[abs(goal[y][x]-exp_goal[y][x]) for x in range(size)] for y in range(size)]
    diff_entro=[[abs(entro_map[y][x]-exp_entro_map[y][x]) for x in range(size)] for y in range(size)]
    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(goal, cmap='gray', vmin=0, vmax=1)  # type: ignore
    axs[0, 0].set_title("Truth") # type: ignore
    axs[1, 0].imshow(entro_map, cmap='gray') # type: ignore
    axs[1, 0].set_title("Entropy") # type: ignore
    axs[1, 1].imshow(exp_entro_map, cmap='gray') # type: ignore
    axs[1, 1].set_title("E[Entropy]") # type: ignore
    axs[0, 1].imshow(exp_goal, cmap='gray', vmin=0, vmax=1) # type: ignore
    axs[0, 1].set_title('Expected Map') # type: ignore
    axs[0, 2].imshow(diff_goal, cmap='gray',vmin=0, vmax=1) # type: ignore
    axs[0, 2].set_title('Pred./Truth') # type: ignore 
    axs[1, 2].imshow(diff_entro, cmap='gray') # type: ignore
    axs[1, 2].set_title('E[Entro]/Entro') # type: ignore
    axs[1, 3].imshow(hits, cmap='gray', vmin=0, vmax=1) # type: ignore
    axs[1, 3].set_title('Hits') # type: ignore
    axs[0, 3].imshow(shots, cmap='gray', vmin=0, vmax=1)  # type: ignore
    axs[0, 3].set_title('Shots') # type: ignore

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(fname)
    plt.close(fig)

def draw_shots(shots:List[List[int]], size:int, fname):
    
    plt.clf()
    plt.imshow([[[x,y] in shots for x in range(size)] for y in range(size)],cmap='gray', vmin=0, vmax=1)
    plt.title("Shots")
    plt.savefig(fname)

def run(run_id:int)->List[int]:
    os.mkdir(f"./img/{run_id}/")
    boards, data = load_data()
    ships:List[List[List[int]]]=data["ships"]  # type: ignore
    size:int=data["size"]  # type: ignore
    goal=get_goal(boards)
    shots=[]
    ys=[]
    n=len(boards)
    i=0
    while n>1:
        print(n)
        entro_map=calculate_entropy_map(boards,size)
        max_entro=max(max(entro_map,key=lambda x: max(x)))
        x,y=choice([(x,y) for x in range(size) for y in range(size) if entro_map[y][x]==max_entro])
        ys.append(n)
        draw_state(boards,goal,shots,entro_map,size,"./img/"+str(run_id)+"/"+str(i)+".png")
        boards=filter_after_shot(boards,goal,x,y)[1]
        shots.append([x,y])
        n=len(boards)
        i+=1
        info=False
        for i2 in range(size):
            for j2 in range(size):
                info |= (not entro_map[i2][j2]==0) # type: ignore
        if not info: 
            n=1
    plt.clf()
    xs=[i for i in range(len(ys))]
    draw_state(boards,goal,shots,[[0 for x in range(size)] for y in range(size)],size,"./img/"+str(run_id)+"/"+str(i)+".png")
    plt.plot(xs,ys)
    plt.title("LogSize")
    plt.yscale("log")
    plt.savefig(f"./img/{run_id}/logsize.png")
    return ys
    
def main():
    if not os.path.exists("boards.json"):
        ships=[[[1,3],[3,1]], [[1,5],[5,1]], [[1,7],[7,1]], [[2,2]]]
        size=8
        save_data(ships,size)
    os.mkdir(f"./replays/")
    yss=[]
    for i in range(50):
        yss.append(run(i))
    plt.clf()
    plt.yscale("log")
    for ys in yss:
        plt.plot([i for i in range(len(ys))],ys)
    plt.title("Lengths")
    plt.savefig("./img/Ts.png")
    with open("runs.json", "w+") as f:
        json.dump(yss,f, indent=3)
    
def myshkin(boards:List[List[List[int]]], shots:List[List[int]], size:int)->Tuple[int,int]:
    entro_map=calculate_entropy_map(boards,size)
    if entro_map==[[0.0 for i in range(size)] for _ in range(size)]:
        exp_map=calculate_expectation_map(boards,size)
        max_exp=max(max(exp_map,key=lambda x: max(x)))
        x,y=choice([(x,y) for x in range(size) for y in range(size) if exp_map[y][x]==max_exp and shots[y][x]==0])
    else:
        max_entro=max(max(entro_map,key=lambda x: max(x)))
        x,y=choice([(x,y) for x in range(size) for y in range(size) if entro_map[y][x]==max_entro])
    return (x,y)

def user_input(boards:List[List[List[int]]], size:int)->Tuple[int,int]:
    x,y = [int(i) for i in input().split(",")]
    return x,y

def play_game(f:Callable[[List[List[List[int]]],List[List[int]], int], Tuple[int,int]],index=-1, show_progress:bool=True):
    boards, data = load_data()
    goal=get_goal(boards)
    ships:List[List[List[int]]]=data["ships"]  # type: ignore
    size:int=data["size"] # type: ignore
    shots=[[0 for i in range(size)] for _ in range(size)]
    board_map=[[0 for i in range(size)] for _ in range(size)]
    replay={"replay":[],"goal":[[x,y] for x in range(size) for y in range(size) if goal[y][x]==1]}
    while(goal!=shots):
        x,y=f(boards,shots,size)
        replay["replay"].append({"board":board_map,"shot":[x,y]})
        boards=filter_after_shot(boards,goal,x,y)[1]
        if goal[y][x]==1:
            shots[y][x]=1
            board_map[y][x]=1
        else:
            board_map[y][x]=-1
        if show_progress:
            pprint(board_map)
    if index==-1:
        with open("./replays/replay_"+datetime.now().time().isoformat().replace(":","").replace(".","")+".json", "w+") as file:
            json.dump(replay,file,indent=3)
    else:
        with open("./replays/replay_"+str(index)+".json", "w+") as file:
            json.dump(replay,file,indent=3)

if __name__ == "__main__":
    for i in range(100):
        play_game(myshkin,i,False)
    