import argparse
import sys
from mnist_original_moe_training import *
from original_moe_training import *
from helper.collect_results import *

expert_layers_types = {'expert_layers': expert_layers}

gate_layers_types = {'gate_layers': gate_layers,
                    'gate_layers_top_k': gate_layers_top_k}

expert_layers_type = expert_layers
gate_layers_type = gate_layers

m = 'mnist_without_reg'
mt = 'moe_expectation_model'
total_experts = 10
k = 0
runs = 1
w_importance_range=[0.0]
w_sample_sim_same_range = [0.0]
w_sample_sim_diff_range = [0.0]
d = 'default_distance_funct'
distance_funct = default_distance_funct
filename = 'mnist_top_k_results.csv'

parser = argparse.ArgumentParser()
parser.add_argument('-e', help='expert layers type')
parser.add_argument('-g', help='gate layers type')
parser.add_argument('-k', help='top k')
parser.add_argument('-m', help='model name')
parser.add_argument('-mt', help='model type')
parser.add_argument('-r', help='number of runs')
parser.add_argument('-E', help='number of epochs')
parser.add_argument('-M', help='number of experts')
parser.add_argument('-D', help='sample distance function')
parser.add_argument('-i', help='Importance factor')
parser.add_argument('-ss', help='sample similarity factor')
parser.add_argument('-sd', help='sample dissimilarity factor')
parser.add_argument('-f', help='filename')
args = vars(parser.parse_args())
print('args', args)

if not args['e'] is None:
    expert_layers_type = expert_layers_types[args['e']] 
if not args['g'] is None:
    gate_layers_type = gate_layers_types[args['g']]    
if not args['k'] is None:
    k = int(args['k'])
if not args['m'] is None:
    m = args['m']
if not args['mt'] is None:
    mt = args['mt']
if not args['r'] is None:
    runs = int(args['r'])
if not args['M'] is None:
    total_experts = int(args['M'])
if not args['E'] is None:
    num_epochs = int(args['E'])
if not args['D'] is None:
    d = args['D']
if not args['i'] is None:
    w_importance_range = [float(args['i'])]
if not args['ss'] is None:
    w_sample_sim_same_range = [float(args['ss'])]
if not args['sd'] is None:
    w_sample_sim_diff_range = [float(args['sd'])]
if not args['f'] is None:
    filename = args['f']

print('expert layers type:', expert_layers_type)
print('gate layers type:', gate_layers_type)
print('k:', k)
print('model name:', m)
print('model type:', mt)
print('runs:', runs)
print('total experts:', total_experts)
print('Num epochs:', num_epochs)
print('importance factor:', w_importance_range[0])
print('sample similarity factor:', w_sample_sim_same_range[0])
print('sample dissimilarity factor:', w_sample_sim_diff_range[0])
print('results filename:', filename)

num_classes = 10

# Paths to where the trained models, figures and results will be stored. You can change this as you see fit.
working_path = './'
model_path = os.path.join(working_path, 'models/mnist')
results_path = os.path.join(working_path, 'results/mnist')

if not os.path.exists(model_path):
    os.mkdir(model_path)


collect_results(m, mt, 
                w_importance_range = w_importance_range,
                w_sample_sim_same_range=w_sample_sim_same_range,
                w_sample_sim_diff_range=w_sample_sim_diff_range,
                total_experts=total_experts, num_classes=num_classes, num_epochs=num_epochs, 
                testloader=testloader, model_path=model_path, results_path=results_path, filename=filename)