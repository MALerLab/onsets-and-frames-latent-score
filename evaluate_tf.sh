output_file="all_output.txt"
# > $output_file  # Clear the file if it exists

for epoch in $(seq 1000 1000 300000); do
    echo "MAPS $epoch" | tee -a $output_file
    python3 evaluate_tf.py /home/sake/userdata/sake/onsets-and-frames-latent-score/runs/transcriber-241213-214302/model-${epoch}.pt --sequence-length 327680 | tee -a $output_file
done


