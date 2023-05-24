using DelimitedFiles
using Plots
using Statistics
using LsqFit

gr() 

# base_path = dirname(pwd())
# for the moment put hte plot manually
base_path = "/home/boris/Documents/Audio_pruning/Audio_pruning/"

N_to_test = [200, 400, 600, 800]
ratios = [0.1, 0.2 ] 
type_prunes = ["hard", "simple", "normal"]
K = 10

Loss_N = zeros(length(N_to_test))
Loss_N_simple = zeros(length(N_to_test), length(ratios))
Loss_N_hard = zeros(length(N_to_test), length(ratios))

for i = 1:(length(N_to_test))
    
    total_path = base_path * "results/history_tiny_prune_$(N_to_test[i])_K$(K)_normal_1.0.csv" 
    Loss_N[i] = mean(readdlm( total_path ,  ',')[:,1])

        for j = 1:length(ratios)

            total_path = base_path * "results/history_tiny_prune_$(N_to_test[i])_K$(K)_simple_$(ratios[j]).csv" 
            Loss_N_simple[i, j] = mean(readdlm( total_path ,  ',')[:,1])

            total_path = base_path * "results/history_tiny_prune_$(N_to_test[i])_K$(K)_hard_$(ratios[j]).csv" 
            Loss_N_hard[i, j ] = mean(readdlm( total_path ,  ',')[:,1])
        end
    
end

scatter(N_to_test, Loss_N, label = "random pruning", marker = :cross)
scatter!(N_to_test, Loss_N_simple[:,2], label = "simple pruning 20%", marker = :xcross)
scatter!(N_to_test, Loss_N_hard[:,2], label = "hard pruning 20%", marker = :utriangle)