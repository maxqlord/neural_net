import numpy as np
import random
import itertools
import math
import matplotlib.pyplot as plt
#classes defined: Percept and Input(extends Percept)
#parameters: the weights (w_ij) and thresholds (t_j)
class Percept:
    def __init__(self, w, t, k):
        self.weights = w
        self.thresh = t
        self.k = k
    def evaluate(self): #returns normal number if input and calculates if perceptron
        input_vals = []
        for i in self.inputs:
            input_vals.append(i.evaluate())
        #print(input_vals)
        return self.sigmoid(np.dot(self.weights, input_vals))
    def evaluate_step(self): #returns normal number if input and calculates if perceptron
        input_vals = []
        for i in self.inputs:
            input_vals.append(i.evaluate_step())
        #print(input_vals)
        return self.step(np.dot(self.weights, input_vals))
    def step(self,t):
        if t > self.thresh:
            return 1
        return 0
    def set_inputs(self, arr):
        self.inputs = arr
    def sigmoid(self, t):
        k = self.k
        #if t > self.thresh:
        #    return 1
        #return 0
        return 1/(1+math.exp(-k*(t-self.thresh)))

#class NN: (4,3,9,2) 5,4,10,3 each layer is NxN matrix  4x3, 3x9, 9x2  eval non recursive matrix multiply > activate
#class NN:
 #   def __init__(
class Input(Percept):
    def __init__(self):
        pass

    def set_value(self,v):
        self.value = v
    def evaluate(self):
        return self.value
    def evaluate_step(self):
        return self.value

def generateResults(size):

    results = ["".join(seq) for seq in itertools.product("01", repeat=size)]

    return results

def createData(size):

    inputs = []

    genInputs = ["".join(seq) for seq in itertools.product("01", repeat=size)]

    for k in genInputs:

        stringList = list(k)
        intList = list(map(int, stringList))
        intList.append(1)
        npArray = np.array(intList)
        inputs.append(npArray)
    createResults = generateResults(2 ** size)
    list1 = []
    entireList = []
    weight_list = []
    for k in range(0, len(createResults)): 
        stringList = list(createResults[k])
        intList = list(map(int, stringList))
        for i in range(0, len(intList)):
            list1.append((inputs[i], intList[i]))
        entireList.append(list1)
        weights = calc_w(list1,size)
        if len(weights) > 0:
            weight_list.append(weights)
        list1 = []
    return weight_list
def createW(size):
    w = np.array([0] * (size + 1))
    return w

def A(t):

    if t > 0:
        return 1
    return 0
def calc_w(list1, size):
    #global correct, incorrect
    #print(list1)
    neural_loop = []
    results = []
    w = createW(size)
    lambdaConst = 1
    program_results = []
    for i in range(0, 100):
        for k in list1:
            f_star = A(np.dot(w, k[0]))
            w += lambdaConst * (k[1] - f_star) * k[0]
    for k in list1:
        program_results.append(A(np.dot(w, k[0])))
        #print(A(np.dot(w, k[0])))
        
    for k in list1:
        results.append(k[1])
    if program_results == results:
        #correct += 1
        #print(w)
        neural_loop.append(w)
        #incorrect += 1
    return neural_loop

def find_xor():
    correct_weights = createData(2)
    x1 = Input()
    x2 = Input()
    #print(correct_weights)
    xor_nodes = []
    #xnor_nodes = []
    xor_nodes_weights = []
    xor_nodes_numbers = []
    perceptrons = []
    for weight in correct_weights:
        perceptrons.append(Percept([weight[0][0], weight[0][1]], weight[0][2]*-1,1))
    #xnor_nodes_weights = []
    #xnor_nodes_numbers = []
    i_counter = 0
    j_counter = 0
    k_counter = 0
    for i in perceptrons:
        
        node_1 = Percept(i.weights, i.thresh,1)
        for j in perceptrons:
            node_2 = Percept(j.weights, j.thresh,1)
            for k in perceptrons:
                #print(i_counter,j_counter,k_counter)
                node_3 = Percept(k.weights, k.thresh,1)
                node_3.set_inputs([node_1,node_2])
                xor = [0,1,1,0]
                xnor = [1,0,0,1]
                arr = []
                node_1.set_inputs([x1,x2])
                node_2.set_inputs([x1,x2])
                for a in range(2):
                    for b in range(2):
                        x1.set_value(a)
                        x2.set_value(b)
                        arr.append(node_3.evaluate_step())
                #print(arr)
                #if(i = 3 and j == 4 and k == 6):
                        #print(i,j,k)
                        #print(arr)
                if(arr == xor):
                    xor_nodes.append([node_1,node_2,node_3])
                    xor_nodes_weights.append([(node_1.weights,node_1.thresh),(node_2.weights,node_2.thresh),(node_3.weights,node_3.thresh)])
                    first = i_counter
                    second = j_counter
                    third = k_counter

                    xor_nodes_numbers.append([first,second,third])
                k_counter+=1
            j_counter+=1
            k_counter = 0
        i_counter+=1
        j_counter = 0
        k_counter = 0
                #if(arr == xnor):
     #               xnor_nodes.append([node_1,node_2,node_3])
     #               xnor_nodes_weights.append([(node_1.weights,node_1.thresh),(node_2.weights,node_2.thresh),(node_3.weights,node_3.thresh)])
     #               xnor_nodes_numbers.append([i,j,k])
    #print(correct_weights)
    print("xor nodes weights: ")
    print(xor_nodes_weights)
    print("xor nodes numbers: ")
    print(xor_nodes_numbers)
    #print("xnor: ")
    #print(xnor_nodes)
def evaluate(self, t, k):
    return 1/(1+math.exp(-k*(t-self.thresh)))
        #if t > self.thresh:
        #    return 1
        #return 0
        
    

def graph_accuracy(acc_array):
    indexArray = []
    for z in range(5,80):
        indexArray.append(z/10.0)
    
    linesAccuracy = plt.plot(indexArray, acc_array)
    plt.ylabel('Percentage Accuracy')
    plt.xlabel('k Value')
    plt.axis([1.0,5.0,50,100])
    plt.setp(linesAccuracy, linewidth=3, color = 'g')
    plt.grid()
    plt.show()
def graph_iterations(itr_array):
    indexArray = []
    for z in range(11,100):
        indexArray.append(z/10.0)
    
    linesAccuracy = plt.plot(indexArray, itr_array)
    plt.ylabel('Iterations')
    plt.xlabel('k Value')
    plt.axis([1.0,10.0,200,1000])
    plt.setp(linesAccuracy, linewidth=3, color = 'g')
    plt.grid()
    plt.show()    
def guess(learning_rate, cutoff):
    c = cutoff
    k = learning_rate #4.65
    x1 = Input()
    x2 = Input()
    weight_1 = [1,0]
    thresh_1 = -1
    weight_2 = [-1,0]
    thresh_2 = -1
    weight_3 = [0, 1]
    thresh_3 = -1
    weight_4 = [0, -1]
    thresh_4 = -1
    weight_AND = [1,1]
    thresh_AND = 1.5
    node_1 = Percept(weight_1, thresh_1,k)
    node_2 = Percept(weight_2, thresh_2,k)
    node_3 = Percept(weight_3, thresh_3,k)
    node_4 = Percept(weight_4, thresh_4,k)
    node_1.set_inputs([x1,x2])
    node_2.set_inputs([x1,x2])
    node_3.set_inputs([x1,x2])
    node_4.set_inputs([x1,x2])
    node_AND1 = Percept(weight_AND, thresh_AND,k)
    node_AND2 = Percept(weight_AND, thresh_AND,k)
    node_AND3 = Percept(weight_AND, thresh_AND,k)
    arr = []
    
    node_AND1.set_inputs([node_1, node_2])
    node_AND2.set_inputs([node_3, node_4])
    node_AND3.set_inputs([node_AND1, node_AND2])
    accuracy_array = []
    
    #for j in range(5, 10):
    
    #print(k)
    correct = 0
    
    for i in range(0, 10000):
        x = random.uniform(-1.5,1.5)
        y = random.uniform(-1.5,1.5)
        #print("coordinates: " + str(x) + ", " + str(y))

        x1.set_value(x)
        x2.set_value(y)
    
        num = node_AND3.evaluate()
        if x**2 + y**2 <= 1 and num >= c:
            correct +=1
        if x**2 + y**2 > 1 and num < c:
            correct +=1
            
    #print(str(k) + "  " + str(100 * (correct/10000)))
    
    return 100 * (correct/10000)
    #graph_accuracy(arr)
def circle():
    arr = []
    avg_arr = []
    c=.5
    for x in range(5, 80):
        k = x/10.0
        for y in range(0,5):
        #counter +=1
            arr.append(guess(k,c))
        average = sum(arr)/len(arr)
        #print(k, arr, average)
        avg_arr.append(average)
        arr = []
    graph_accuracy(avg_arr)
def search(k):
    learning_rate = .1
    target = .001
    weights = np.array([0,0,0,0,0,0,0,0,0])
    def randomize():
        arr = []
        for i in range(len(weights)):
            arr.append(random.uniform(-1,1))
        return arr
    def result(first, second,k):
        one = 1/(1+math.exp(-k*(first*weights[0]+second*weights[1]-weights[2])))
        two = 1/(1+math.exp(-k*(first*weights[3]+second*weights[4]-weights[5])))
        final = 1/(1+math.exp(-k*(one*weights[6]+two*weights[7]-weights[8])))
        return final
    def err(k):
        return (result(0,0,k)-0)**2 + (result(0,1,k)-1)**2 + (result(1,0,k)-1)**2 + (result(1,1,k)-0)**2
    weights = np.array(randomize())
    error = 4
    counter = 0
    restart_counter  = 0
    while error > target:

        if(counter > 1000):
            restart_counter +=1
            if(restart_counter > 5):
                return counter
            counter = 0
            weights = np.array(randomize())
            error = 4
        counter +=1
        #print(counter,error)
        change = np.multiply(np.array(randomize()),learning_rate)
        #print(change)
        old = np.copy(weights)
        weights = np.add(weights, change)
        new_err = err(k)
        #print("new",new_err)
        if error > new_err:
            error = new_err
        else:
            weights = old
    #print(error)
    #print(weights)
    return counter
def check_search():
    total_arr = []
    for z in range(11,100): #k values
        #print(z)
        k = z/10.0
        avg_array = []
        for i in range (0, 100): #average
            #print(i)
            iterations = search(k)
            avg_array.append(iterations)
        #print(avg_array)
        average = sum(avg_array)/len(avg_array)
        #print(average)
        total_arr.append(average)
        avg_array = []
    print(total_arr)
    #graph_iterations(total_arr)

find_xor()
circle()
#print(guess(4.65,.5))
#search(7.5)
#check_search()









