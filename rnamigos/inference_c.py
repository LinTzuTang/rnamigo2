import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import time
from torch.utils.data import DataLoader
import pandas as pd

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.learning.dataset import InferenceDataset
from rnamigos.utils.graph_utils import get_dgl_graph
from rnamigos.learning.models import get_model_from_dirpath
import argparse
from rnaglib.prepare_data import fr3d_to_graph


def inference(
    dgl_graph,
    smiles_list,
    out_path="rnamigos_out.csv",
    mixing_coeffs=(0.5, 0.0, 0.5),
    models_path=None,
    dump_all=False,
):
    """
    Run inference from python objects
    """
    # Load models
    script_dir = os.path.dirname(__file__)
    if models_path is None:
        models_path = {
            "dock": os.path.join(script_dir, "../results/trained_models/dock/dock_42"),
            "native_fp": os.path.join(
                script_dir, "../results/trained_models/native_fp/fp_42"
            ),
            "is_native": os.path.join(
                script_dir, "../results/trained_models/is_native/native_42"
            ),
        }
    models = {
        model_name: get_model_from_dirpath(model_path)
        for model_name, model_path in models_path.items()
    }

    # Get ready to loop through ligands
    dataset = InferenceDataset(smiles_list)
    batch_size = 64
    loader_args = {
        "shuffle": False,
        "batch_size": batch_size,
        "num_workers": 0,
        "collate_fn": dataset.collate,
    }
    dataloader = DataLoader(dataset=dataset, **loader_args)

    results = {model_name: [] for model_name in models.keys()}
    t0 = time.time()
    for i, (ligands_graph, ligands_vector) in enumerate(dataloader):
        for model_name, model in models.items():
            if model_name == "native_fp":
                scores = model.predict_ligands(dgl_graph, ligands_vector)
            else:
                scores = model.predict_ligands(dgl_graph, ligands_graph)
            scores = list(scores[:, 0].numpy())
            results[model_name].extend(scores)
        if not i % 10 and i > 0:
            print(f"Done {i * batch_size}/{len(dataset)} in {time.time() - t0}")

    # Post-process raw scores to get a consistent 'higher is better' numpy array score
    for model_name, all_scores in results.items():
        all_scores = np.asarray(all_scores)
        # Flip raw scores as lower is better for those models
        if model_name in {"dock", "native_fp"}:
            all_scores = -all_scores
        results[model_name] = all_scores

    # Normalize each methods outputs and mix methods together : best mix = 0.44, 0.39, 0.17
    def normalize(scores):
        out_scores = (scores - scores.min()) / (scores.max() - scores.min())
        return out_scores

    normalized_results = {
        model_name: normalize(result) for model_name, result in results.items()
    }
    mixed_scores = (
        mixing_coeffs[0] * normalized_results["dock"]
        + mixing_coeffs[1] * normalized_results["native_fp"]
        + mixing_coeffs[2] * normalized_results["is_native"]
    )

    rows = []
    if not dump_all:
        for smiles, score in zip(smiles_list, mixed_scores):
            rows.append({"smiles": smiles, "score": score})
    else:
        for smiles, dock_score, native_score, fp_score, mixed_score in zip(
            smiles_list,
            results["dock"],
            results["is_native"],
            results["native_fp"],
            mixed_scores,
        ):
            rows.append(
                {
                    "smiles": smiles,
                    "dock_score": dock_score,
                    "native_score": native_score,
                    "fp_score": fp_score,
                    "mixed_score": mixed_score,
                }
            )
    result_df = pd.DataFrame(rows)
    if out_path is not None:
        result_df.to_csv(out_path)
    return result_df


def do_inference(cif_path, residue_list, ligands_path, out_path, dump_all=False):
    """
    Run inference from files
    """
    # Get dgl graph with node expansion BFS
    dgl_graph = get_dgl_graph(cif_path, residue_list)
    print("Successfully built the graph")
    smiles_list = [s.lstrip().rstrip() for s in list(open(ligands_path).readlines())]
    print("Successfully parsed ligands, ready for inference")
    inference(
        dgl_graph=dgl_graph,
        smiles_list=smiles_list,
        out_path=out_path,
        dump_all=dump_all,
    )


@hydra.main(version_base=None, config_path="conf", config_name="inference")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Done importing")
    do_inference(
        cif_path=cfg.cif_path,
        residue_list=cfg.residue_list,
        ligands_path=cfg.ligands_path,
        out_path=cfg.out_path,
    )


def get_residue_list(cif_path, chain=None):
    # Convert CIF file to networkx graph
    nx_graph = fr3d_to_graph(cif_path)

    # Filter based on the chain provided (default is None, which returns all nodes)
    if chain:
        node_list = [
            node.split(".")[1] + "." + node.split(".")[2]
            for node in nx_graph.nodes()
            if node.split(".")[1] == chain
        ]
    else:
        node_list = [
            node.split(".")[1] + "." + node.split(".")[2] for node in nx_graph.nodes()
        ]

    # Return as a comma-separated string
    return ",".join(node_list)


def log_time(start_time, end_time, seed, chain, base_path):
    elapsed_time = end_time - start_time
    log_path = os.path.join(base_path, "inference_times.log")
    
    with open(log_path, "a") as log_file:
        # log_file.write(f"Seed: {seed}, Chain: {chain}, Time: {elapsed_time:.2f} seconds\n")
        log_file.write(f"Chain: {chain}, Time: {elapsed_time:.2f} seconds\n")
        

def calculate_ranks(out_path_all, smiles_str):
    # Generate the file names
    file = '1M_all.out'
    
    # Initialize an empty list to store results
    results = []
    
   
    # df = pd.read_csv(os.path.join(dir, file), index_col=0)
    df = pd.read_csv(out_path_all, index_col=0)
    # Sort by mixed_score
    df_sorted_dock = df.sort_values(by='mixed_score', ascending=False).reset_index(drop=True)
    
    # Find the dock rank
    try:
        dock_rank = df_sorted_dock[df_sorted_dock['smiles'] == smiles_str].index[0] + 1  # 1-based
        results.append([file, dock_rank])
    except IndexError:
        print(f"SMILES string not found in file: {file}")
    
    # Create a DataFrame with the results
    df_results = pd.DataFrame(results, columns=['file', 'rank'])
    
    # Calculate average rank and top percentage
    average_dock_rank = df_results['rank'].mean()
    df_results['top_percent'] = (df_results['rank'] / 1000001) * 100
    average_dock_top_percent = df_results['top_percent'].mean()
    
    # Append the averages as a row
    df_results.loc['Average'] = ['Average', average_dock_rank, average_dock_top_percent]
    
    return df_results



def main(pdbid):
    # Define the base path
    base_path = f"case_study_{pdbid}"

    # Define the CIF file path
    # cif_path = "case_study/1ibm.cif"
    cif_path = f"./{base_path}/{pdbid}.cif"
    # cif_path = "case_study/5uzz.cif"

    # Get residue lists for all and chain 'B'
    residue_list_all = get_residue_list(cif_path)
    # residue_list_B = get_residue_list(cif_path, chain="B")

    # Loop through seeds 1 to 5
    # for seed in range(1, 6):
        # Run inference for all residues
    ligands_path = (
        f"{base_path}/Chemspace_Screening_Compounds_SMILES_sampled1M_.smi"
    )
    # out_path_all = f"{base_path}/1M_all.out"
    out_path_all = f"{base_path}/1M1_all.out"

        # Time the inference for all residues
    start_time = time.time()
    do_inference(
        cif_path=cif_path,
        residue_list=residue_list_all.split(","),
        ligands_path=ligands_path,
        out_path=out_path_all,
        dump_all=True,
    )
    
    with open(ligands_path, 'r') as f:
        content = f.readlines()
        smiles_str = content[-1].strip()
    
    # smiles_str = '[H]/N=C(\\c1ccc2c(c1)c(cc(n2)C)Nc3cccc(c3)OC)/N'
    results_df = calculate_ranks(out_path_all, smiles_str)
    results_df.to_csv(f"{base_path}/results_.csv")
    
    end_time = time.time()

        # Log the time taken for the "all" run
    log_time(start_time, end_time, '0', "1M", base_path)

        # # Run inference for chain 'B' residues
        # out_path_B = f"{base_path}/seed{seed}_B.out"

        # # Time the inference for chain 'B'
        # start_time = time.time()
        # do_inference(
        #     cif_path=cif_path,
        #     residue_list=residue_list_B.split(","),
        #     ligands_path=ligands_path,
        #     out_path=out_path_B,
        #     dump_all=True,
        # )
        # end_time = time.time()

        # # Log the time taken for the "B" run
        # log_time(start_time, end_time, seed, "B", base_path)


if __name__ == "__main__":
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Run inference on specified PDB ID.")
    parser.add_argument(
        "--pdbid", required=True, help="The PDB ID for the case study (e.g., 9cpd)."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the pdbid argument
    main(args.pdbid)
    
# if __name__ == "__main__":
#     # Define the base path
#     base_path = f"case_study_9cpd"

#     # Define the CIF file path
#     cif_path = f"./{base_path}/9cpd.cif"

#     # Get residue lists for all and chain 'B'
#     residue_list_all = get_residue_list(cif_path)
    
#     ligands_path = (
#             f"{base_path}/Chemspace_Screening_Compounds_SMILES_sampled_seed1.smi"
#         )
#     out_path_all = f"seed1_all.out"
#     do_inference(
#             cif_path=cif_path,
#             residue_list=residue_list_all.split(","),
#             ligands_path=ligands_path,
#             out_path=out_path_all,
#             dump_all=True,
#         )