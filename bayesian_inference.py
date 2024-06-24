import re
from copy import deepcopy
import random
import sys

# Function to find common elements between two lists
def common_l(l1,l2):
        l_common = []
        for i in l1:
            if i in l2:
                l_common.append(i)
        return l_common

# Function to split a string based on a delimiter and filter out empty values
def split_function(delimiter,st):
        split_values = re.split(delimiter,st)
        filtered_values = [value for value in split_values if value]

        return list(map(int,filtered_values))

# Class for Bayesian Network
class Bayesian_Network:
    def __init__(self,network_file):
        self.network_file = network_file

    # Function to produce truth values for given variables
    def produce_truth_values(self,l):
        res = [l]
        curr = 0
        while (curr<len(l)):
            for i in range(pow(2,curr)):
                curr1 = res.pop(0)
                res.append(curr1[:curr] + [curr1[curr]] + curr1[curr+1:])
                res.append(curr1[:curr] + [-curr1[curr]] + curr1[curr+1:])

            curr += 1
        return res

    # Function to read the Bayesian network from a file
    def read_b_file(self):
        d_sup = dict()
        s_list = []
        with open(self.network_file,'r') as file:

            n = -1
            curr_truth_values = []
            while True:
                line = file.readline()
                if not line:
                    break

                if (n==-1):
                    n = int(line)
                elif (len(curr_truth_values)==0):
                    l = list(map(int,line.split()))
                    s = ""
                    s += str(l[0])
                    if(len(l)!=1):
                        s += ":"
                    for i in range(1,len(l)):
                        if (i!=len(l)-1):
                            s += str(l[i]) + ","
                        else:
                            s += str(l[i])
                    
                    s_list.append(s)
                    d_sup[s] = dict()
                    curr_truth_values += self.produce_truth_values(l[1:]+[l[0]])
                else:
                    t1 = curr_truth_values.pop(0)
                    t2 = curr_truth_values.pop(0)
                    t1.sort()
                    t2.sort()
                    p1,p2 = map(float,line.split())
                    d_sup[s_list[-1]][tuple(t1)] = p1
                    d_sup[s_list[-1]][tuple(t2)] = p2

        return n,d_sup,s_list

# Class for variable elimination
class variable_elimination:

    def __init__(self,n,q_l,e_l,d_init):
        self.n = n
        self.q_l = q_l
        self.e_l = e_l
        self.d_init = d_init

    # Function to identify hidden variables
    def hidden_variables(self,n,q,e):
        l = set()
        hidden = []
        for i in q:
            l.add(abs(i))
        for i in e:
            l.add(abs(i))
        l1 = list(l)
        l1.sort()
        for i in range(1,n+1):
            if l1 and l1[0]==i:
                l1.remove(i)
            else:
                hidden.append(i)
        return hidden

    # Function to reduce tables
    def reduce(self):
        keys = list(self.d_init.keys())
        d_new = dict()
        e_var_new = []
        for e in self.e_l:
            e_var_new.append(-e)

        for k in keys:
            d_new[k] = dict()
            curr_table = self.d_init[k]

            for k1 in curr_table:
                l1 = list(k1)
                remove = False
                for i in l1:
                    if i in e_var_new:
                        remove = True
                        break
                
                if(remove==False):
                    d_new[k][k1] = curr_table[k1]
        
        return d_new


    # Function to find keys containing a specific variable
    def find_keys_with_h(self,key_list,h):

        keys_containing_h = []

        for k in key_list:
            l = split_function(':|,',k)
            if h in l:
                keys_containing_h.append(k)
        return keys_containing_h

    # Function to create a string representation of variables i.e. of the form q:e
    def make_string(self,q,e):
        s = ""
        for i in range(len(q)-1):
            s += str(q[i]) + ","
        if len(q)!=0:
            s += str(q[len(q)-1]) + ":"

        for i in range(len(e)-1):
            s += str(e[i]) + ","
        if len(e)!=0:
            s += str(e[len(e)-1])

        return s


    # Function to join two strings representing variables
    def join_string(self,s1,s2):
        split1 = re.split(':',s1)
        split2 = re.split(':',s2)
        q1 = split_function(',',split1[0])
        if (len(split1)==1):
            e1 = []
        else:
            e1 = split_function(',',split1[1])
        q2 = split_function(',',split2[0])
        if (len(split2)==1):
            e2 = []
        else:
            e2 = split_function(',',split2[1])

        common_q1_e2 = common_l(q1,e2)
        for i in common_q1_e2:
            q1.remove(i)
            e2.remove(i)

        common_q2_e1 = common_l(q2,e1)
        for i in common_q2_e1:
            q2.remove(i)
            e1.remove(i)

        E1 = self.combine_lists(e1,e2)
        Q1 = list(set(q1+q2+common_q1_e2+common_q2_e1))
        s = self.make_string(Q1,E1)

        return s

    # Function to combine two lists
    def combine_lists(self,l1,l2):
        l = set(l1 + l2)
        return list(l)
    
    # Function to join tables
    def join(self,d_init,keys_to_join):
        d_new = dict()

        if len(keys_to_join)<=1:
            return d_init,keys_to_join[0]
        else:
            for x in d_init:
                if x not in keys_to_join:
                    d_new[x] = deepcopy(d_init[x])

            curr_table_key = keys_to_join.pop(0)
            curr_table_key_list = split_function(':|,',curr_table_key)

            d_final = dict()
            d_final[curr_table_key] = deepcopy(d_init[curr_table_key])

            while len(keys_to_join)!=0:
                next_table_key = keys_to_join.pop(0)
                next_table_key_list = split_function(':|,',next_table_key)

                common_rv = common_l(curr_table_key_list,next_table_key_list)
            
                new_key = self.join_string(curr_table_key,next_table_key)

                if new_key not in d_final:
                    d_final[new_key] = dict()

                for l_curr in d_final[curr_table_key]:
                    l1_curr = list(l_curr)

                    for l_next in d_init[next_table_key]:
                        l1_next = list(l_next)

                        common_l1 = common_l(l1_curr,l1_next)
                        if(len(common_l1)==len(common_rv)):
                            l = self.combine_lists(l1_curr,l1_next)
                            d_final[new_key][tuple(l)] = d_final[curr_table_key][l_curr] * d_init[next_table_key][l_next]

                curr_table_key = deepcopy(new_key)
                curr_table_key_list = split_function(':|,',curr_table_key)

            d_new[curr_table_key] = deepcopy(d_final[curr_table_key])

            return d_new,curr_table_key


    # Function to sum out a variable from a table
    def Sum(self,curr_key_dict,key,varia_to_sum): # here key is the string

        d_cur = deepcopy(curr_key_dict)

        d_upd = dict()
        split1 = re.split(':',key)
        q1 = split_function(',',split1[0])

        remove_true = False
        if len(q1)==1 and q1[0]==varia_to_sum:
            remove_true = True
        if(len(split1)==1):
            e1 = []
        else:
            e1 = split_function(',',split1[1])
        
        if varia_to_sum in q1:
            q1.remove(varia_to_sum)
        if varia_to_sum in e1:
            e1.remove(varia_to_sum)

        key_1 = self.make_string(q1,e1)
        keys_list = list(d_cur.keys())

        while (len(keys_list)!=0):
            k1 = list(keys_list.pop(0))
            for kk2 in keys_list:
                k2 = list(kk2)
                comm_l = common_l(k1,k2)
                combi = k1+k2
                if(len(comm_l)==len(k1)-1 and (varia_to_sum in combi) and (-varia_to_sum in combi)):
                    prob = d_cur[tuple(k1)] + d_cur[tuple(kk2)]
                    if varia_to_sum in k1:
                        k1.remove(varia_to_sum)            
                    if -varia_to_sum in k1:
                        k1.remove(-varia_to_sum)
                    if remove_true==True:
                        return d_upd,-1
                    d_upd[tuple(k1)] = prob
                    keys_list.remove(kk2)
                    break
        
        return d_upd,key_1


    # Function to eliminate hidden variables from tables
    def eliminate_hidden_variables(self,d_new,hidden_l):

        d_curr = deepcopy(d_new)

        for h in hidden_l:
            keys_to_join = self.find_keys_with_h(list(d_curr.keys()),h)

            d_curr,key_added = self.join(d_curr,keys_to_join)

            d_upd,key_1 = self.Sum(d_curr[key_added],key_added,h)

            del d_curr[key_added]

            if key_1!=-1:
                d_curr[key_1] = deepcopy(d_upd)

        return d_curr


    # Function to join remaining tables
    def join_remaining(self,d_curr):
        
        keys_to_join = list(d_curr.keys())
        d_new,curr_table_key = self.join(d_curr,keys_to_join)
        return d_new


    # HERE INSTEAD OF MAKING SEPARATE NORMALISE FUNCTION, I AM NORMALISING IT INSIDE THE BELOW FUNCTION 
    # THUS, NORMALISE FUNCTION IS THE SAME AS BELOW

    # Function to calculate probability using variable elimination
    def variable_elimination_prob(self,n,q_l,e_l,d_init):
        
        hidden_l = self.hidden_variables(n,q_l,e_l)

        d_new1 = self.reduce()
    
        d_new2 = self.eliminate_hidden_variables(d_new1,hidden_l)
    
        d_new3 = self.join_remaining(d_new2)

        key = list(d_new3.keys())[0]

        d_new4 = deepcopy(d_new3[key])

        key_pri = list(d_new3.keys())[0]
        
        p1 = -1
        p2 = -1

        # sum will return directly the table
        if (len(e_l)==0):
            p2 = 1
        else:
            for q in q_l:
                d_new4,key = self.Sum(d_new4,key,abs(q))  
        
        combine_l_e = self.combine_lists(q_l,e_l)

        for ele in d_new3[key_pri]:
            current_ele = list(ele)
            if len(common_l(combine_l_e,current_ele))==len(current_ele):
                p1 = d_new3[key_pri][ele]
                break
        
        if p2==-1:
            for elem in d_new4:
                current_ele = list(elem)
                if len(common_l(e_l,current_ele))==len(current_ele):
                    p2 = d_new4[elem]
                    break
        
        if p1==0 or p1==-1:
            return 0
        return p1/p2


# Class for rejection sampling
class Rejection_Sampling:

    def __init__(self,d_init,e_l,n):
        self.d_init = d_init
        self.map_dict = {}
        self.e_l = e_l
        self.n = n
        self.bayesian_graph = {}

    # Function to create nodes and map them to variables
    def create_nodes(self,keys_list):
        for node in keys_list:
            split = split_function(':|,',node)
            self.map_dict[int(split[0])] = node

            self.bayesian_graph[int(split[0])] = []


    # Function to create a dictionary mapping child nodes to parent nodes
    def make_child_dict(self,n):
        keys_list = list(self.d_init.keys())
        child_dict = dict()
        for i in range(1,n+1):
            child_dict[i] = []

        for key in keys_list:
            child_list = split_function(':|,',key)
            q = child_list[0]
            if len(child_list)>1:
                e_l = child_list[1:]
            else:
                e_l = []

            for i in e_l:
                child_dict[i].append(q)

        return child_dict


    # Function to perform topological sort
    def topologicalSort(self,graph):
        h = deepcopy(graph)
        n = len(graph)
        top_list = []

        while len(h)!=0:
            v = -1
            for key in h:
                if len(h[key])==0:
                    v = key
                    break

            top_list.insert(0,v)
            del h[v]
            for key in h:
                if v in h[key]:
                    h[key].remove(v)
        
        return top_list


    # Function to generate a random sample
    def random_sample(self):
        return random.random()


    # Function to generate samples using rejection sampling
    def generate_samples(self,d_init,e_l,n):
        
        keys_list = list(self.d_init.keys())
        self.create_nodes(keys_list)
        byesian_graph = self.make_child_dict(n)
        topological_list = self.topologicalSort(byesian_graph)
        sample_list = []

        for i in range(100000):
            curr_sample = []

            for j in range(n):
                topo = topological_list[j]
                curr_table = d_init[self.map_dict[topo]]
                random_prob = random.random()

                t1 = []
                t2 = []
                for key1 in curr_table:
                    key = list(key1)
                    if(t1==[] and len(common_l(key,curr_sample))==len(key)-1):
                        t1 = key
                        p1 = curr_table[key1]
                    elif (t2==[] and len(common_l(key,curr_sample))==len(key)-1):
                        t2 = key
                        p2 = curr_table[key1]
                        break
                
                if topo in t1:
                    if random_prob<=p1:
                        curr_sample.append(topo)
                    else:
                        curr_sample.append(-topo)
                else:
                    if random_prob<=p2:
                        curr_sample.append(topo)
                    else:
                        curr_sample.append(-topo)

            if(len(common_l(curr_sample,e_l))==len(e_l)):
                sample_list.append(curr_sample)

        return sample_list


    # Function to perform rejection sampling
    def rejection_sampling(self,d_init,e_l,q_l,n):

        sample_list = self.generate_samples(d_init,e_l,n)

        deno = len(sample_list)
        nume = 0
        numerator = []
        for sample in sample_list:
            if(len(common_l(q_l,sample))==len(q_l)):
                nume += 1
                numerator.append(sample)

        if deno==0:
            return 0
        prob = nume/deno
        
        return prob


# Function to read and solve queries from input files
def read_and_solve_query(b_file,q_file):

    bay_network = Bayesian_Network(b_file)

    n,d_sup,s_list = bay_network.read_b_file()

    with open(q_file, "r") as file:
        line = file.readline()

        while line:

            words = line.split()

            q_list = []
            e_list = []
            q_true = False
            e_true = False

            for word in words:
                if word == "q":
                    q_true = True
                    e_true = False
                elif word == "e":
                    q_true = False
                    e_true = True
                elif q_true == True:
                    if(word[0]=="~"):
                        q_list.append(-int(word[1:]))
                    else:
                        q_list.append(int(word))
                    
                elif e_true == True:
                    if(word[0]=="~"):
                        e_list.append(-int(word[1:]))
                    else:
                        e_list.append(int(word))

            if words[0]=="ve":
                variable_elimina = variable_elimination(n,q_list,e_list,d_sup)

                variable_elimination_probabity = variable_elimina.variable_elimination_prob(n,q_list,e_list,d_sup)
                print("Query : ",line.split('\n')[0])
                print("Probability by variable elination method is : ",variable_elimination_probabity)

            else:
                rejection_sampling1 = Rejection_Sampling(d_sup,e_list,n)
                rejection_sampling_prob = rejection_sampling1.rejection_sampling(d_sup,e_list,q_list,n)
                print("Query : ",line.split('\n')[0])
                print("Probability by rejection sampling method is : ",rejection_sampling_prob)

            line = file.readline()

# Read input file paths from command line arguments and solve queries
b_file = sys.argv[1]
q_file = sys.argv[2]

read_and_solve_query(b_file,q_file)


# l = []

# for i in range(100):
#     l.append(read_and_solve_query(q_file))

# print("Max : ",max(l))
# print("Min : ",min(l))
# print("Average : ",sum(l)/100)


        





