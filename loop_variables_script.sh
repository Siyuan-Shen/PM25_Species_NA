#!/bin/bash

# Define the range for the loop
#variables=('GC_PM25' 'GC_NH4' 'GC_SO4' 'GC_SOA'  'GC_OM' 'GC_BC' 'GC_DST' 'GC_SSLT'
#           'NH3_anthro_emi' 'NO_anthro_emi' 'OC_anthro_emi' 'BC_anthro_emi' 
#            'DST_offline_emi' 'PBLH' 'RH' 'T2M' 'U10M' 'V10M' 'PRECTOT'
#            'Urban_Builtup_Lands' 'Lat' 'Lon' 'elevation' 'Month_of_Year' 'Population'
#            ) # variables for excludion

variables=('GC_NIT' 'GC_OC' 'Croplands' 'Crop_Nat_Vege_Mos' 'Permanent_Wetlands' 'SO2_anthro_emi' 'NMVOC_anthro_emi' 'Crop_Nat_Vege_Mos' 'Permanent_Wetlands' 'SSLT_offline_emi' 
           'major_roads' 'minor_roads' 'motorway' 'primary' 'secondary' 'trunk' 'unclassified' 'residential' 
           'major_roads_dist' 'minor_roads_dist' 'motorway_dist' 'primary_dist' 'secondary_dist' 'trunk_dist' 'unclassified_dist' 'residential_dist')
#            'DST_offline_emi' 'PBLH' 'RH' 'T2M' 'U10M' 'V10M' 'PRECTOT'
#            'Urban_Builtup_Lands' 'Lat' 'Lon' 'elevation' 'Population'
#            ) # variables for inclusion


# Job script file
job_script="run_gpu.bsub"

# Print the total number of iterations
total_iterations=${#variables[@]}
echo "Total number of iterations: $total_iterations"

# Loop through the variables
for ((i=0; i<total_iterations; i++)); do
    var=${variables[i]}

    # Print the current iteration number
    echo "Iteration $((i+1)) of $total_iterations: Processing variable $var"
    
    # Create a temporary modified script
    modified_script="modified_job_script_${i}.bsub"
    cp $job_script $modified_script

    # Update the pause_time calculation
    sed -i "s/^pause_time=\$((RANDOM % 50 .*/pause_time=\$((RANDOM % 10 + (${i} * 120)))/" $modified_script
    # Use sed to replace variables in the script (exclusion test)
    sed -i "s/^var=.*/var=${var}/" $modified_script
    #sed -i "s/^Exclude_Variables_Sensitivity_Test_Switch=.*/Exclude_Variables_Sensitivity_Test_Switch=false/" $modified_script
    #sed -i "s/^Exclude_Variables_Sensitivity_Test_Variables=.*/Exclude_Variables_Sensitivity_Test_Variables=[['${var}']]/" $modified_script
    sed -i "s/^#BSUB -J .*/#BSUB -J \"1.8.0 include ${var}\"/" $modified_script

    # Submit the modified script using bsub
    echo "Submitting job for variable $var..."
    bsub < $modified_script

    # Pause for 150 seconds before the next submission
    echo "Waiting for 2 seconds before the next job..."
    sleep 2

    # Clean up temporary script after submission
    rm $modified_script
done