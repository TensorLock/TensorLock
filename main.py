import os
import subprocess
import sys
import argparse

def run_script(script_path, description):
    """
    Runs a python script in its directory context.
    """
    abs_path = os.path.abspath(script_path)
    directory = os.path.dirname(abs_path)
    script_name = os.path.basename(abs_path)
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting: {description}")
    print(f"📂 Directory: {directory}")
    print(f"📜 Script: {script_name}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(abs_path):
        print(f"❌ Error: Script not found at {abs_path}")
        return False

    original_cwd = os.getcwd()
    try:
        os.chdir(directory)        
        # Run the script and capture output to stdout/stderr
        result = subprocess.run([sys.executable, script_name], check=False)
        
        if result.returncode == 0:
            print(f"\n✅ Finished: {description} ")
            return True
        else:
            print(f"\n❌ Failed: {description} (Exit Code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    parser = argparse.ArgumentParser(description="TensorLock Pipeline Runner")
    parser.add_argument("--skip-download", action="store_true", help="Skip the model download step")
    parser.add_argument("--start-from", type=int, default=0, help="Start from a specific step (0-6)")
    args = parser.parse_args()

    # Define the pipeline steps
    # Format: (Step Number, Script Path, Description)
    pipeline = []

    # Step 0: Ground Truth
    pipeline.append((0, "code/step0_ground_truth/gen_ground_truth_matrix.py", "Generate Ground Truth Matrix"))

    # Step 1: Clustering
    if not args.skip_download:
        pipeline.append((1, "code/step1_cluster/0-download.py", "Step 1.0: Download Models"))
    
    pipeline.append((1, "code/step1_cluster/1-preprocess.py", "Step 1.1: Preprocess Models"))
    pipeline.append((1, "code/step1_cluster/2-dequantize.py", "Step 1.2: Dequantize Models"))
    pipeline.append((1, "code/step1_cluster/3-fingerprints.py", "Step 1.3: Generate Fingerprints"))
    pipeline.append((1, "code/step1_cluster/4-cluster.py", "Step 1.4: Cluster Models"))

    # Step 2: Quantization Identification
    pipeline.append((2, "code/step2_quantize/1-qtz_identify.py", "Step 2: Identify Quantization Relations"))

    # Step 3: Merge Identification
    pipeline.append((3, "code/step3_merge/1-merge_identify.py", "Step 3: Identify Merge Relations"))

    # Step 4: PEFT Identification
    pipeline.append((4, "code/step4_peft/1-peft_identify.py", "Step 4: Identify PEFT Relations"))

    # Step 5: Finetune Identification
    pipeline.append((5, "code/step5_finetune/1-cal_entropy.py", "Step 5.1: Calculate Entropy"))
    pipeline.append((5, "code/step5_finetune/2-ft_identify.py", "Step 5.2:  Identify Finetune Relations"))
    pipeline.append((5, "code/step5_finetune/3-MST.py", "Step 5.3: Minimum Spanning Tree (Finalize)"))

    # Step 6: Evaluation
    pipeline.append((6, "code/step6_eval/eval_cluster.py", "Step 6.1: Evaluate Clustering"))
    pipeline.append((6, "code/step6_eval/eval_graph.py", "Step 6.2: Evaluate Graph"))

    # Execute pipeline
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    for step_num, script_rel_path, description in pipeline:
        if step_num < args.start_from:
            continue
            
        script_path = os.path.join(base_dir, script_rel_path)
        success = run_script(script_path, description)
        
        if not success:
            print(f"\n⛔ Pipeline stopped due to failure in step: {description}")
            sys.exit(1)

    print(f"\n{'='*80}")
    print(f"🎉 Pipeline completed successfully!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
