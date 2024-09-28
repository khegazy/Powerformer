import numpy as np 

results = {}
with open('result.txt', 'r') as file:
    for line in file:
        if 'Patch' in line:

            items = line.split('_')
            data = items[3]
            if data not in results:
                results[data] = {}
            pred_len = items[7][2:]
            if pred_len not in results[data]:
                results[data][pred_len] = {}
            decay_type = items[16][2:]
            time_decay = items[17]
            if decay_type == 'None':
                time_decay = 0
            elif 'rain' in time_decay:
                decay_type = 'remove'
                time_decay = 0
            time_decay = float(time_decay)
                
            print(data, decay_type, time_decay)
            ln_idx = 1
            """
            idx = line.find('_at')
            line = line[idx+3:]
            print("new line", line)
            print(items)
            """
        elif line == '\n':
            continue
        else:
            items = line.split(',')
            print("items", len(items), items)
            values = []
            for itm in items:
                idx = itm.find(':')
                metric = itm[:idx].strip()
                print("itm", itm)
                val = float(itm[idx+1:])
                print(metric, val)
                values.append((metric, val))

            results[data][pred_len][time_decay] = values

print(results.keys())

for data in results.keys():
    print(data)
    with open(f'table_total_{data}.tex', 'w') as output:
        for pl, pl_dict in sorted(results[data].items()):
            out_str = f"& {pl}"
            print(pl)
            for tm, tm_list in sorted(pl_dict.items()):
                print('\t',tm)
                out_str += f" & {tm_list[0][1]:0.4} / {tm_list[1][1]:0.4}"
            output.write(out_str + "  \\\\ \n")
    
    with open(f'table_mse_{data}.tex', 'w') as output:
        for pl, pl_dict in sorted(results[data].items()):
            out_str = f"& {pl}"
            print(pl)
            for tm, tm_list in sorted(pl_dict.items()):
                print('\t',tm)
                out_str += f" & {tm_list[0][1]:0.4}"
            output.write(out_str + "  \\\\ \n")

    with open(f'table_mae_{data}.tex', 'w') as output:
        for pl, pl_dict in sorted(results[data].items()):
            out_str = f"& {pl}"
            print(pl)
            for tm, tm_list in sorted(pl_dict.items()):
                print('\t',tm)
                out_str += f" & {tm_list[1][1]:0.4}"
            output.write(out_str + "  \\\\ \n")

