import subprocess

# this is only for this particular example. 
N_to_test = [200, 400, 600, 800]
# N_to_test = [5000, 6000,  7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 30000, 42700, 51300, 59800, 68400, 76900]
ratios = [0.1, 0.2 , 0.3] 
type_prunes = ['hard', 'simple', 'normal']
K = 10

for j in range(100):

    # #For N scan 
    for i in N_to_test:
        print('Run with N = {}'.format(i))

        for t in type_prunes:
            print('Run with pruning = {}'.format(t))

            if t == 'normal':
                cmd_str = 'python Classification_pruning.py --N_samples {} --K {} --type_prune {} --ratio {}'.format(i, K, t , 1)    
                subprocess.run(cmd_str, shell=True)
            else:
                for k in ratios:
                    print('Run with ratio = {}'.format(k))
                    cmd_str = 'python Classification_pruning.py --N_samples {} --K {} --type_prune {} --ratio {}'.format(i, K, t , k)
                    subprocess.run(cmd_str, shell=True)


# #For K scan
# K_to_test = [5 , 15 , 25 , 35 , 45 , 55 ,65 , 75 , 85, 95, 105, 155, 250]
# K_to_test = [155, 250]

# for i in K_to_test:
#    print('Run with K = {}'.format(i))
#    cmd_str = 'python Classification_pruning_KWS.py --N_samples {} --K {} --type_prune {} --ratio {}'.format(68000, i, type_prune , ratio)
#    # subprocess.run(cmd_str, shell=True)
   
#    for j in range(100):
#        print('Run {} / 100'.format(j+1))
#        subprocess.run(cmd_str, shell=True)
