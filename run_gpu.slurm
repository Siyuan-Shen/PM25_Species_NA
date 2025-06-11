#!/bin/bash
#SBATCH --job-name="replicate_padding_Penalty1_16_Epoch71_layer0_311_avgpooling_exclude_SGbiases_Elu_LastBlockTanh"
#SBATCH --partition=general
#SBATCH --mem=150G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --exclude=c2-gpu-[009]
#SBATCH --output=job_output/job-%j-output.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.siyuan@wustl.edu
#SBATCH --account=compute-rvmartin
#SBATCH -vvv


# If you're using containers with Slurm
# Uncomment this line if needed:
#SBATCH --container-image=syword/python3-pytorch:2024.05
#SBATCH --container-mounts=/storage1/fs1/rvmartin/Active/s.siyuan:/my-projects,/storage1/fs1/rvmartin2/Active/s.siyuan:/my-projects2
#SBATCH --container-workdir=/my-projects/Projects/PM25_Speices_DL_2023/code/Training_Evaluation_Estimation/PM25/v1.8.2

# Optional: mimic host exclusion if needed via constraints or scheduler filters

# Run your code
cd /my-projects/Projects/PM25_Speices_DL_2023/code/Training_Evaluation_Estimation/PM25/v1.8.2
echo "Entered the directory ......."

# Random pause

/bin/true
pause_time=$((RANDOM % 50 + 0))
echo "Pausing for $pause_time seconds..."
sleep $pause_time

var='PM25'
# sed -i '/\[Sensitivity_Test\-Settings\.Training_Settings]/,/^\[/{/Exclude_Variables_Sensitivity_Test_Switch/s/=.*/=true/;}' config.toml
sed -i '/\[Sensitivity_Test\-Settings\.Training_Settings]/,/^\[/{/Exclude_Variables_Sensitivity_Test_Variables/s/=.*/=[['\"${var}\"']]/;}' config.toml
sed -i '/\[Sensitivity_Test\-Settings\.Training_Settings]/,/^\[/{/Include_Variables_Sensitivity_Test_Variables/s/=.*/=[['\"${var}\"']]/;}' config.toml


Buffer_size=[80]
sed -i '/\[BLCO\-CrossValidation\]/,/^\[/{/Buffer_size/s/=.*/= '"$Buffer_size"'/}' config.toml
sed -i '/\[Uncertainty\-Settings\.BLISCO_LOWESS_Uncertainty_Settings\]/,/^\[/{/Buffer_radii_forUncertainty/s/=.*/= '"$Buffer_size"'/}' config.toml

beginyears_endyears=[2023]
Estimation_years=[[2023]]
sed -i '/\[Estimation\-Settings\.Training_Settings\]/,/^\[/{/beginyears/s/=.*/= '"$beginyears_endyears"'/;/endyears/s/=.*/= '"$beginyears_endyears"'/;}' config.toml
sed -i '/\[Estimation\-Settings\.Map_Estimation_Settings\]/,/^\[/{/Estiamtion_trained_beginyears/s/=.*/= '"$beginyears_endyears"'/}' config.toml
sed -i '/\[Estimation\-Settings\.Map_Estimation_Settings\]/,/^\[/{/Estiamtion_trained_endyears/s/=.*/= '"$beginyears_endyears"'/}' config.toml
sed -i '/\[Estimation\-Settings\.Map_Estimation_Settings\]/,/^\[/{/Estimation_years/s/=.*/= '"$Estimation_years"'/}' config.toml
sed -i '/\[Estimation\-Settings\.Visualization_Settings\]/,/^\[/{/Map_Plot_YEARS/s/=.*/= '"$beginyears_endyears"'/}' config.toml

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Configuration updated. Elapsed time for configuration update: $elapsed_time seconds."


python3 main.py

