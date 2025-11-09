#!/usr/bin/env python3
"""
VMEvalKit Question Creation - Flexible Task Generation

This script provides flexible question/task generation with customizable domain
selection and quantities. Generate questions for specific reasoning domains
or create comprehensive datasets across all domains.

Key Features:
- Generate questions for specific domains (chess, maze, raven, rotation, sudoku)

Output Structure:
- Each question gets its own folder: data/questions/{domain}_task/{question_id}/
- Contains: first_frame.png, final_frame.png, prompt.txt, question_metadata.json
- Master dataset JSON with complete metadata and organization

Author: VMEvalKit Team
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vmevalkit.runner.dataset import (
    create_vmeval_dataset_direct, 
    read_dataset_from_folders, 
    print_dataset_summary
)
from vmevalkit.utils.constant import DOMAIN_REGISTRY

def main():
    """Flexible VMEvalKit question creation with customizable options."""

    parser = argparse.ArgumentParser(
        description="VMEvalKit Question Creation - Flexible task generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Generate 50 questions per domain (default)
            python create_questions.py
            
            # Generate specific number per domain
            python create_questions.py --pairs-per-domain 100
            
            # Generate for specific domains only
            python create_questions.py --task chess maze --pairs-per-domain 25
            
            # Download arc_agi_2 tasks from HuggingFace
            python create_questions.py --task arc_agi_2
            
            # Generate small test set
            python create_questions.py --task chess --pairs-per-domain 5
            
            # Just read and analyze existing questions
            python create_questions.py --read-only
            
            # Use different random seed
            python create_questions.py --random-seed 123 --pairs-per-domain 10
        """
    )
    
    parser.add_argument(
        "--pairs-per-domain", 
        type=int, 
        default=50, 
        help="Number of task pairs to generate per domain (default: 50)"
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducible generation (default: 42)"
    )
    
    parser.add_argument(
        "--read-only", 
        action="store_true", 
        help="Only read and analyze existing dataset from folders, don't generate new questions"
    )
    
    parser.add_argument(
        "--task", 
        nargs='+', 
        choices=list(DOMAIN_REGISTRY.keys()), 
        help=f"Specific task domain(s) to generate. Available: {', '.join(DOMAIN_REGISTRY.keys())}. If not specified, generates all domains."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/questions",
        help="Output directory for questions (default: data/questions)"
    )
    
    parser.add_argument(
        "--list-domains",
        action="store_true", 
        help="List all available task domains and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_domains:
        print("🧠 Available Task Domains:")
        print("=" * 60)
        for domain_key, domain_info in DOMAIN_REGISTRY.items():
            hf_info = " (HuggingFace)" if domain_info.get('hf', False) else ""
            print(f"{domain_info.get('emoji', '🔹')} {domain_key:15} - {domain_info['description']}{hf_info}")
        print(f"\nTotal: {len(DOMAIN_REGISTRY)} reasoning domains available")
        print("\nUse --task to select specific domains, or run without --task for all domains.")
        return

    if args.read_only:
        print("=" * 70)
        print("📂 Reading existing questions from folder structure...")
        print(f"   Reading from: {args.output_dir}")
        dataset = read_dataset_from_folders(Path(args.output_dir))
        print_dataset_summary(dataset)
        print("=" * 70)
        return

    output_path = Path(args.output_dir)
    selected_domains = args.task if args.task else list(DOMAIN_REGISTRY.keys())
    
    hf_domains = [d for d in selected_domains if DOMAIN_REGISTRY.get(d, {}).get('hf', False) is True]
    regular_domains = [d for d in selected_domains if DOMAIN_REGISTRY.get(d, {}).get('hf', False) is not True]
    
    if hf_domains:
        for domain in hf_domains:
            domain_info = DOMAIN_REGISTRY[domain]
            print("=" * 70)
            print(f"📥 Downloading {domain} tasks from HuggingFace...")
            print("=" * 70)
            print(f"   Dataset: {domain_info.get('hf_dataset')}")
            print(f"   Subset: {domain_info.get('hf_subset')}")
            print(f"   Split: {domain_info.get('hf_split')}")
            print(f"📁 Output directory: {args.output_dir}")
            
            from datasets import load_dataset
            
            hf_dataset_name = domain_info.get('hf_dataset')
            hf_subset = domain_info.get('hf_subset')
            hf_split = domain_info.get('hf_split', 'train')
            
            print(f"   Loading dataset: {hf_dataset_name}")
            if hf_subset:
                dataset = load_dataset(hf_dataset_name, hf_subset, split=hf_split)
            else:
                dataset = load_dataset(hf_dataset_name, split=hf_split)
            
            hf_domain = domain_info.get('hf_domain', domain)
            task_id_prefix = domain_info.get('hf_task_id_prefix', domain)
            prompt_column = domain_info.get('hf_prompt_column', 'prompt')
            image_column = domain_info.get('hf_image_column', 'image')
            solution_image_column = domain_info.get('hf_solution_image_column', 'solution_image')
            
            tasks = []
            for idx, item in enumerate(dataset):
                task_id = f"{task_id_prefix}_{idx:04d}"
                
                prompt = item.get(prompt_column, "")
                first_image = item.get(image_column)
                solution_image = item.get(solution_image_column)
                
                if not prompt:
                    print(f"      ⚠️  Skipping {task_id}: Missing prompt")
                    continue
                
                if first_image is None:
                    print(f"      ⚠️  Skipping {task_id}: Missing image")
                    continue
                
                task = {
                    "id": task_id,
                    "domain": hf_domain,
                    "prompt": prompt,
                    "first_image": first_image,
                    "solution_image": solution_image
                }
                
                tasks.append(task)
            
            domain_dir = output_path / f"{domain_info.get('hf_domain', domain)}_task"
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            import json
            from PIL import Image
            
            downloaded_tasks = []
            for task in tasks:
                task_id = task['id']
                task_dir = domain_dir / task_id
                task_dir.mkdir(parents=True, exist_ok=True)
                
                first_image = task['first_image']
                if not isinstance(first_image, Image.Image):
                    first_image = Image.fromarray(first_image) if hasattr(first_image, 'shape') else Image.open(first_image)
                if first_image.mode != "RGB":
                    first_image = first_image.convert("RGB")
                
                dest_first = task_dir / "first_frame.png"
                first_image.save(dest_first, format="PNG")
                
                solution_image = task.get('solution_image')
                final_image_path = None
                if solution_image is not None:
                    if not isinstance(solution_image, Image.Image):
                        solution_image = Image.fromarray(solution_image) if hasattr(solution_image, 'shape') else Image.open(solution_image)
                    if solution_image.mode != "RGB":
                        solution_image = solution_image.convert("RGB")
                    
                    dest_final = task_dir / "final_frame.png"
                    solution_image.save(dest_final, format="PNG")
                    final_image_path = str(Path(f"{task['domain']}_task") / task_id / "final_frame.png")
                
                prompt_file = task_dir / "prompt.txt"
                prompt_file.write_text(task['prompt'])
                
                task_metadata = {
                    "id": task_id,
                    "domain": task['domain'],
                    "prompt": task['prompt'],
                    "first_image_path": str(Path(f"{task['domain']}_task") / task_id / "first_frame.png"),
                    "final_image_path": final_image_path,
                    "created_at": datetime.now().isoformat() + 'Z',
                    "source": domain_info.get('hf_dataset'),
                    "subset": domain_info.get('hf_subset')
                }
                
                metadata_file = task_dir / "question_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(task_metadata, f, indent=2, default=str)
                
                downloaded_tasks.append(task_metadata)
            
            print(f"✅ Downloaded {len(downloaded_tasks)} {domain} tasks to {domain_dir}")
            print()
    
    if regular_domains:
        total_questions = len(regular_domains) * args.pairs_per_domain
        
        print("=" * 70)
        print("🚀 VMEvalKit Question Generation Plan")
        print("=" * 70)
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🎯 Selected domains: {', '.join(regular_domains)} ({len(regular_domains)} domains)")
        print(f"📊 Questions per domain: {args.pairs_per_domain}")
        print(f"🔢 Total questions to generate: {total_questions}")
        print(f"🎲 Random seed: {args.random_seed}")
        print()

        dataset, questions_dir = create_vmeval_dataset_direct(
            pairs_per_domain=args.pairs_per_domain, 
            random_seed=args.random_seed,
            selected_tasks=regular_domains
        )
        
        print_dataset_summary(dataset)
    
    print("=" * 70)
    print("📂 Reading all questions from folder structure...")
    print("=" * 70)
    final_dataset = read_dataset_from_folders(output_path)
    print_dataset_summary(final_dataset)
    
    print(f"📁 Questions saved in: {output_path}")
    print(f"🔗 Per-question folders: {output_path}/<domain>_task/<question_id>/")
    print()
    print("🎉 VMEvalKit Questions ready for video generation!")
    print("🚀 Next steps:")
    print(f"   • Generate videos: python examples/generate_videos.py")
    print(f"   • Score videos: python examples/score_videos.py human")
    print("=" * 70)

if __name__ == "__main__":
    main()
