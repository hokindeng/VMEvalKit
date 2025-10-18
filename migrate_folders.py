#!/usr/bin/env python3
"""
Migrate flat inference folders to nested structure.

This script migrates incorrectly placed flat folders like:
  data/outputs/pilot_experiment/openai-sora-2_rotation_0006_20251018_020844/

To the correct nested structure:
  data/outputs/pilot_experiment/openai-sora-2/rotation_task/rotation_0006/openai-sora-2_rotation_0006_20251018_020844/
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


def extract_model_and_task_info(folder_name: str, metadata: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Extract model name, domain, and task_id from folder name and metadata.
    
    Args:
        folder_name: Name of the folder (e.g., "openai-sora-2_rotation_0006_20251018_020844")
        metadata: Metadata dictionary from metadata.json
        
    Returns:
        Dictionary with model_name, domain, task_id, or None if extraction fails
    """
    # Extract from metadata if available
    question_data = metadata.get("question_data", {})
    domain = question_data.get("domain")
    task_id = question_data.get("id")
    
    # Try to extract model name from folder name
    # Pattern: {model_name}_{task_id}_{timestamp}
    # The model name should match the inference metadata
    model_name = metadata.get("inference", {}).get("model")
    
    # If model is "sora-2" from metadata, convert to full model name
    if model_name == "sora-2":
        model_name = "openai-sora-2"
    elif model_name == "sora-2-pro":
        model_name = "openai-sora-2-pro"
    
    # If we still don't have model_name, try to extract from folder
    if not model_name:
        parts = folder_name.split("_")
        if folder_name.startswith("openai-sora-"):
            model_name = "_".join(parts[:3])  # openai-sora-2
        else:
            # For other models, assume model name is first part before task_id
            model_name = parts[0]
    
    if not domain or not task_id:
        print(f"  ⚠️  Warning: Could not extract full info from {folder_name}")
        print(f"     domain={domain}, task_id={task_id}")
        return None
    
    return {
        "model_name": model_name,
        "domain": domain,
        "task_id": task_id
    }


def update_metadata_paths(metadata_file: Path, old_base: str, new_base: str) -> None:
    """
    Update all paths in metadata.json from old to new base path.
    
    Args:
        metadata_file: Path to metadata.json
        old_base: Old base path to replace
        new_base: New base path
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Update paths recursively
    def update_paths(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: update_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [update_paths(item) for item in obj]
        elif isinstance(obj, str) and old_base in obj:
            return obj.replace(old_base, new_base)
        return obj
    
    updated_metadata = update_paths(metadata)
    
    # Write back
    with open(metadata_file, 'w') as f:
        json.dump(updated_metadata, f, indent=2)


def migrate_folder(
    source_folder: Path,
    output_base: Path,
    dry_run: bool = False
) -> bool:
    """
    Migrate a single flat folder to nested structure.
    
    Args:
        source_folder: Path to flat folder to migrate
        output_base: Base output directory (e.g., data/outputs/pilot_experiment)
        dry_run: If True, only print what would be done
        
    Returns:
        True if migration was successful or would be successful
    """
    folder_name = source_folder.name
    
    print(f"\n📦 Processing: {folder_name}")
    
    # Read metadata
    metadata_file = source_folder / "metadata.json"
    if not metadata_file.exists():
        print(f"  ⚠️  No metadata.json found, skipping")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Extract info
    info = extract_model_and_task_info(folder_name, metadata)
    if not info:
        return False
    
    model_name = info["model_name"]
    domain = info["domain"]
    task_id = info["task_id"]
    
    # Construct target path
    domain_dir = f"{domain}_task"
    target_path = output_base / model_name / domain_dir / task_id / folder_name
    
    print(f"  ℹ️  Model: {model_name}")
    print(f"  ℹ️  Domain: {domain} → {domain_dir}")
    print(f"  ℹ️  Task ID: {task_id}")
    print(f"  📍 From: {source_folder}")
    print(f"  📍 To:   {target_path}")
    
    if dry_run:
        print(f"  🔍 [DRY RUN] Would move folder and update paths")
        return True
    
    # Create target directory structure
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if target already exists
    if target_path.exists():
        print(f"  ⚠️  Target already exists: {target_path}")
        print(f"  ⚠️  Skipping to avoid overwrite")
        return False
    
    # Move folder
    print(f"  📦 Moving folder...")
    shutil.move(str(source_folder), str(target_path))
    
    # Update metadata paths
    print(f"  📝 Updating metadata paths...")
    new_metadata_file = target_path / "metadata.json"
    old_path_str = str(source_folder.relative_to(output_base.parent))
    new_path_str = str(target_path.relative_to(output_base.parent))
    update_metadata_paths(new_metadata_file, old_path_str, new_path_str)
    
    print(f"  ✅ Migration complete!")
    return True


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate flat inference folders to nested structure")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs/pilot_experiment",
        help="Base output directory (default: data/outputs/pilot_experiment)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )
    args = parser.parse_args()
    
    output_base = Path(args.output_dir)
    
    if not output_base.exists():
        print(f"❌ Output directory not found: {output_base}")
        return
    
    print("🔍 Scanning for flat folders to migrate...")
    print(f"   Base directory: {output_base}")
    
    if args.dry_run:
        print("   🔍 DRY RUN MODE - No files will be modified")
    
    # Find all flat folders (those with timestamps directly in pilot_experiment/)
    flat_folders = []
    empty_folders = []
    
    for item in output_base.iterdir():
        if item.is_dir() and "_20" in item.name:
            # Check if folder is empty or has no video files
            metadata_file = item / "metadata.json"
            video_dir = item / "video"
            
            if not metadata_file.exists():
                # Check if it's empty
                has_videos = False
                if video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
                    has_videos = len(video_files) > 0
                
                if not has_videos:
                    empty_folders.append(item)
                    continue
            
            # Likely a timestamped folder at wrong level
            flat_folders.append(item)
    
    # Handle empty folders first
    if empty_folders:
        print(f"\n🗑️  Found {len(empty_folders)} empty/failed folder(s):\n")
        for folder in sorted(empty_folders):
            print(f"  - {folder.name}")
        
        if args.dry_run:
            print("\n  🔍 [DRY RUN] Would delete these empty folders")
        else:
            if not args.yes:
                response = input("\n⚠️  Delete empty folders? [y/N]: ")
                if response.lower() != 'y':
                    print("  Skipping empty folder deletion")
                    return
            
            for folder in empty_folders:
                shutil.rmtree(folder)
                print(f"  🗑️  Deleted: {folder.name}")
            print(f"\n✅ Deleted {len(empty_folders)} empty folder(s)")
    
    if not flat_folders:
        print("\n✅ No flat folders found to migrate!")
        return
    
    print(f"\n📋 Found {len(flat_folders)} folder(s) to migrate:\n")
    for folder in sorted(flat_folders):
        print(f"  - {folder.name}")
    
    if not args.dry_run and not args.yes:
        response = input("\n⚠️  Proceed with migration? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Migration cancelled")
            return
    
    # Migrate each folder
    print("\n" + "="*80)
    print("STARTING MIGRATION")
    print("="*80)
    
    success_count = 0
    failed_count = 0
    
    for folder in sorted(flat_folders):
        try:
            if migrate_folder(folder, output_base, dry_run=args.dry_run):
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("MIGRATION SUMMARY")
    print("="*80)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed:     {failed_count}")
    
    if args.dry_run:
        print("\n🔍 This was a DRY RUN. Run without --dry-run to apply changes.")
    else:
        print("\n🎉 Migration complete!")


if __name__ == "__main__":
    main()

