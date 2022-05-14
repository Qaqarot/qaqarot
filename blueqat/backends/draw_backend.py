from .backendbase import Backend
from ..circuit import Circuit
from ..gate import *

# To avoid ImportError, don't import this here.
# import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
import math

class DrawCircuit(Backend):
    """Backend for draw output."""
    
    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        # Lazy import to avoid unneeded ImportError.

        qlist = {}
        flg = 0
        time = 0
        add_edge = []
        remove_edge = []

        for i in range(n_qubits):
            qlist[i] = [{'num': flg, 'gate': 'q'+str(i), 'angle': '', 'xpos': 0, 'ypos': i, 'type': 'qubit'}]
            flg += 1
        
        time += 1
        return gates, (qlist, n_qubits, [flg], [time], add_edge, remove_edge)

    def _postprocess_run(self, ctx):
        import networkx as nx
        import matplotlib.pyplot as plt
        
        #color_code
        color_gate = {}
        color_gate['X'] = color_gate['Y'] = color_gate['Z'] = '#0BB0E2'
        color_gate['RX'] = color_gate['RY'] = color_gate['RZ'] = '#FCD000'
        color_gate['H'] = color_gate['T'] = color_gate['S'] = '#E6000A'
        color_gate['M'] = 'white'
        
        qlist = ctx[0]
        n_qubits = ctx[1]
        flg = ctx[2][-1]
        time = ctx[3][-1]
        
        #measurement
        for i in range(n_qubits):
            qlist[i].append({'num': flg, 'gate': 'M', 'angle': '', 'xpos': 30, 'ypos': i + math.floor((time-1)/30)*(n_qubits+1), 'type': 'measurement'})
            flg += 1
        
        G = nx.Graph()

        for i in range(n_qubits):
            for j in range(len(qlist[i])-1):
                G.add_edge(qlist[i][j]['num'], qlist[i][j+1]['num'])
        
        #twoqubit connections
        for item in ctx[4]:
            G.add_edge(item[0], item[1])

        for item in ctx[5]:
            G.remove_edge(item[0], item[1])

        #image size
        plt.figure(1, figsize=(30, (n_qubits+1)*(math.floor(time/30)+1)), dpi=60)

        labels = {}
        colors = {}
        angles = {}
        sizes = {}

        for i in range(n_qubits):
            for j in range(len(qlist[i])):
                angles[qlist[i][j]['num']] = qlist[i][j]['angle']
                labels[qlist[i][j]['num']] = qlist[i][j]['gate']
                sizes[qlist[i][j]['num']] = 1200
                if qlist[i][j]['type'] == 'dummy':
                    colors[qlist[i][j]['num']] = 'white'
                    sizes[qlist[i][j]['num']] = 0
                elif qlist[i][j]['gate'] == '' or qlist[i][j]['gate'] == 'CZ':
                    colors[qlist[i][j]['num']] = 'black'
                    sizes[qlist[i][j]['num']] = 100
                elif qlist[i][j]['type'] == 'qubit':
                    colors[qlist[i][j]['num']] = 'white'
                else:
                    colors[qlist[i][j]['num']] = color_gate[qlist[i][j]['gate']]

        
        #position
        pos = {}
        for i in range(n_qubits):
            for j in range(len(qlist[i])):
                pos[qlist[i][j]['num']] = (qlist[i][j]['xpos'], (n_qubits+1)*(math.floor(time/30)+1) - qlist[i][j]['ypos'])

        #dummy qubit just for top and bottom margin
        labels[flg]= ''
        colors[flg] = 'black'
        sizes[flg] = 0
        pos[flg] = (0, (n_qubits+1)*(math.floor(time/30)+1)+1)
        G.add_node(flg)
        labels[flg+1]= ''
        colors[flg+1] = 'black'
        sizes[flg+1] = 0
        pos[flg+1] = (0, 1)
        G.add_node(flg+1)
       
        nx.set_node_attributes(G, labels, 'label')
        nx.set_node_attributes(G, colors, 'color')
        nx.set_node_attributes(G, angles, 'angle')
        nx.set_node_attributes(G, sizes, 'size')

        options = {
            "font_size": 12,
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
        }

        node_labels = nx.get_node_attributes(G, 'label')
        node_colors = [colors[i] for i in nx.get_node_attributes(G, 'color')]
        node_sizes = [sizes[i] for i in nx.get_node_attributes(G, 'size')]
        nx.draw_networkx(G, pos, labels = node_labels, node_color = node_colors, node_size = node_sizes, **options)

        #label positions for angles
        pos_attrs = pos.copy()
        for i in pos_attrs:
            pos_attrs[i] = (pos_attrs[i][0]+0.4, pos_attrs[i][1]-0.4)
    
        node_attrs = nx.get_node_attributes(G, 'angle')
        custom_node_attrs = {}

        for node, attr in node_attrs.items():
            custom_node_attrs[node] = attr

        nx.draw_networkx_labels(G, pos_attrs, labels = custom_node_attrs, font_size=9)
        #plt.axis('off')
        plt.show()
        
        return 

    def _one_qubit_gate_noargs(self, gate, ctx):
        flg = ctx[2][-1]
        time = ctx[3][-1]
        qlist = ctx[0]
        
        time_adjust = time%30
        if time_adjust == 0:
            for i in range(ctx[1]):
                ypos_adjust = i + (math.floor(time/30)-1)*(ctx[1]+1)
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 30, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
            time += 1
            
            for i in range(ctx[1]):
                ypos_adjust = i + math.floor(time/30)*(ctx[1]+1)
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 0, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
                ctx[5].append((flg-1, flg-1-ctx[1]))
        
        time_adjust = time%30
        for idx in gate.target_iter(ctx[1]):
            ypos_adjust = idx + math.floor(time/30)*(ctx[1]+1)
            qlist[idx].append({'num': flg, 'gate': gate.lowername.upper(), 'angle': '', 'xpos': time_adjust, 'ypos': ypos_adjust, 'type': 'gate'})
            flg += 1
        ctx[2].append(flg)
        ctx[3].append(time+1)
        return ctx

    gate_x = gate_y = gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_t = gate_tdg = _one_qubit_gate_noargs
    gate_s = gate_sdg = _one_qubit_gate_noargs
    
    def _one_qubit_gate_args_theta(self, gate, ctx):
        flg = ctx[2][-1]
        time = ctx[3][-1]
        qlist = ctx[0]
        
        time_adjust = time%30
        if time_adjust == 0:
            for i in range(ctx[1]):
                ypos_adjust = i + (math.floor(time/30)-1)*(ctx[1]+1)
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 30, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
            time += 1
            
            for i in range(ctx[1]):
                ypos_adjust = i + math.floor(time/30)*(ctx[1]+1)
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 0, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
                ctx[5].append((flg-1, flg-1-ctx[1]))
        
        time_adjust = time%30
        for idx in gate.target_iter(ctx[1]):
            ypos_adjust = idx + math.floor(time/30)*(ctx[1]+1)
            qlist[idx].append({'num': flg, 'gate': gate.lowername.upper(), 'angle': round(gate.theta, 2), 'xpos': time_adjust, 'ypos': ypos_adjust, 'type': 'gate'})
            flg += 1
        ctx[2].append(flg)
        ctx[3].append(time+1)
        return ctx

    gate_rx = _one_qubit_gate_args_theta
    gate_ry = _one_qubit_gate_args_theta
    gate_rz = _one_qubit_gate_args_theta
    gate_phase = _one_qubit_gate_args_theta

    def gate_i(self, gate, ctx):
        time = ctx[3][-1]
        ctx[3].append(time+1)
        return ctx
    
    def _two_qubit_gate_noargs(self, gate, ctx):
        flg = ctx[2][-1]
        time = ctx[3][-1]
        qlist = ctx[0]
        
        tg = ''
        if gate.lowername == 'cx':
            tg = 'x'
        elif gate.lowername == 'cy':
            tg = 'y'
        elif gate.lowername == 'cz':
            tg = 'z'

        time_adjust = time%30
        if time_adjust == 0:
            for i in range(ctx[1]):
                ypos_adjust = i + (math.floor(time/30)-1)*(ctx[1]+1)
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 30, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
            time += 1
            
            for i in range(ctx[1]):
                ypos_adjust = i + math.floor(time/30)*(ctx[1]+1)
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 0, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
                ctx[5].append((flg-1, flg-1-ctx[1]))

        time_adjust = time%30        
        for control, target in gate.control_target_iter(ctx[1]):
            qlist[target].append({'num': flg, 'gate': tg.upper(), 'angle': '', 'xpos': time_adjust, 'ypos': target + math.floor(time/30)*(ctx[1]+1), 'type': 'gate'})
            flg += 1
            qlist[control].append({'num': flg, 'gate': '', 'angle': '', 'xpos': time_adjust, 'ypos': control + math.floor(time/30)*(ctx[1]+1), 'type': 'gate'})
            flg += 1
            ctx[4].append((flg-2, flg-1))
        ctx[2].append(flg)
        ctx[3].append(time+1)
        return ctx
    
    gate_cx = gate_cy = gate_cz = _two_qubit_gate_noargs

    def _three_qubit_gate_noargs(self, gate, ctx):
        return ctx

    gate_ccx = _three_qubit_gate_noargs
    gate_cswap = _three_qubit_gate_noargs

    def gate_measure(self, gate, ctx):
        return ctx

    gate_reset = _one_qubit_gate_noargs
