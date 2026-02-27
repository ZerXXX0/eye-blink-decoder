"""
Pipeline Testing Suite for Eye-Blink Morse Code System
=======================================================

Streamlit-based interactive testing for each component:
1. NLP Text Correction (IndoBERT Seq2Seq)
2. YOLO Eye State Classification
3. EAR (Eye Aspect Ratio) Analysis
4. Full Pipeline Integration Test

Run with: streamlit run test_pipelines.py

Date: January 2026
"""

import streamlit as st
import cv2
import numpy as np
import time
import sys
import os
import pandas as pd
from typing import Optional, Dict, Any, List
from collections import deque
import glob
import zipfile
import shutil
import plotly.express as px
import plotly.graph_objects as go

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from implementation
from implementation import (
    EyeAnalyzer,
    YOLOEyeClassifier,
    ConfidenceFusion,
    IndoBERTCorrector,
    RuleBasedCorrector,
    SystemConfig,
    EyeState,
    YOLOResult,
    EyeData,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    preprocess_for_yolo,
)


# =============================================================================
# ROBOFLOW DATASET IMPORT
# =============================================================================

def render_roboflow_import_section():
    """Render Roboflow dataset import section."""
    st.subheader("Import Test Set from Roboflow")
    st.markdown("**Note:** This feature supports downloading datasets from Roboflow for YOLO and EAR testing.")
    
    with st.expander("🔽 Import Roboflow Dataset", expanded=False):
        # Input fields for Roboflow dataset
        col1, col2 = st.columns([3, 1])
        
        with col1:
            roboflow_code = st.text_area(
                "Paste Roboflow Download Code:",
                placeholder="""# Example Roboflow download code:
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
version = project.version(1)
dataset = version.download("folder")  # Use "folder", "clip", or other supported format""",
                height=140,
                help="Paste the complete download code snippet from your Roboflow project. Make sure to use the correct download format (e.g., 'folder' for classification, 'yolov8' for object detection)"
            )
        
        with col2:
            st.markdown("**Dataset Type:**")
            dataset_type = st.radio(
                "Select type:",
                ["YOLO Classification", "EAR Testing", "Both"],
                help="Choose the purpose for this dataset"
            )
        
        # Configuration options
        st.markdown("**Import Settings:**")
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            extract_to = st.text_input(
                "Extract to folder:",
                value="roboflow_datasets",
                help="Local folder to extract the dataset"
            )
        
        with config_col2:
            max_images = st.number_input(
                "Max images to use:",
                min_value=10,
                max_value=10000,
                value=500,
                help="Maximum number of images to use for testing"
            )
        
        with config_col3:
            use_test_split = st.checkbox(
                "Use test split only",
                value=True,
                help="Only use test split if available"
            )
        
        # Import button
        if st.button("Import Roboflow Dataset", type="primary", use_container_width=True):
            if roboflow_code.strip():
                import_roboflow_dataset(roboflow_code, extract_to, dataset_type, max_images, use_test_split)
            else:
                st.error("Please paste the Roboflow download code first.")
        
        # Show current datasets
        if os.path.exists("roboflow_datasets"):
            st.markdown("**Available Imported Datasets:**")
            datasets = [d for d in os.listdir("roboflow_datasets") if os.path.isdir(os.path.join("roboflow_datasets", d))]
            if datasets:
                for dataset in datasets:
                    dataset_path = os.path.join("roboflow_datasets", dataset)
                    image_count = count_images_in_dataset(dataset_path)
                    st.markdown(f"- `{dataset}` ({image_count} images)")
            else:
                st.info("No datasets imported yet.")


def import_roboflow_dataset(code_snippet: str, extract_to: str, dataset_type: str, max_images: int, use_test_split: bool):
    """Import dataset from Roboflow using the provided code snippet."""
    try:
        # Create extraction directory
        os.makedirs(extract_to, exist_ok=True)
        
        with st.spinner("Importing Roboflow dataset..."):
            # Progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Install roboflow if needed
                status_text.text("Checking Roboflow installation...")
                progress_bar.progress(0.1)
                
                try:
                    import roboflow
                    st.info("✅ Roboflow package is installed")
                except ImportError:
                    status_text.text("Installing Roboflow package...")
                    st.info("Installing Roboflow package...")
                    os.system("pip install roboflow")
                    import roboflow
                    st.success("✅ Roboflow package installed")
                
                # Step 2: Extract download format from code snippet
                status_text.text("Parsing download code...")
                progress_bar.progress(0.2)
                
                # Extract download format from the user's code
                download_format = "folder"  # Default fallback
                lines = code_snippet.split('\n')
                for line in lines:
                    if '.download(' in line and '"' in line:
                        # Extract format from download() call
                        start_quote = line.find('"') + 1
                        end_quote = line.find('"', start_quote)
                        if start_quote > 0 and end_quote > start_quote:
                            download_format = line[start_quote:end_quote]
                            break
                
                st.info(f"Using download format: `{download_format}`")
                
                # Step 3: Execute the download code with enhanced debugging
                status_text.text("Setting up Roboflow connection...")
                progress_bar.progress(0.3)
                
                # Create a controlled execution environment
                local_vars = {}
                global_vars = {'Roboflow': roboflow.Roboflow}
                
                # Show current working directory before execution
                original_cwd = os.getcwd()
                st.info(f"Current working directory: {original_cwd}")
                st.info(f"Target download directory: {extract_to}")
                
                try:
                    # Execute the user's code line by line for better debugging
                    st.info("Executing Roboflow code...")
                    st.code(code_snippet, language="python")
                    
                    exec(code_snippet, global_vars, local_vars)
                    st.success("✅ Roboflow authentication successful")
                    
                    # Find the dataset variable (usually 'dataset' or 'version')
                    dataset = None
                    dataset_var_name = None
                    for var_name, var_value in local_vars.items():
                        if hasattr(var_value, 'download') or str(type(var_value)).endswith('VersionClass\'>'):
                            dataset = var_value
                            dataset_var_name = var_name
                            break
                    
                    if dataset is None:
                        # Show available variables for debugging
                        available_vars = list(local_vars.keys())
                        st.error(f"Could not find dataset object. Available variables: {available_vars}")
                        raise ValueError("Could not find dataset object in the provided code. Make sure to assign the result to a variable (e.g., 'dataset = project.version(...)')")
                    
                    st.success(f"✅ Found dataset object: `{dataset_var_name}`")
                    
                    # Step 4: Download the dataset with comprehensive debugging
                    status_text.text("Downloading dataset from Roboflow...")
                    progress_bar.progress(0.5)
                    
                    # Change to the target directory for download
                    os.chdir(extract_to)
                    st.info(f"Changed to download directory: {extract_to}")
                    
                    # List directory contents before download
                    before_download = set(os.listdir(extract_to)) if os.path.exists(extract_to) else set()
                    st.info(f"Directory contents before download: {list(before_download) if before_download else 'empty'}")
                    
                    # Perform download
                    st.info(f"Calling dataset.download('{download_format}')...")
                    download_result = dataset.download(download_format)
                    
                    # Check what happened after download
                    after_download = set(os.listdir(extract_to)) if os.path.exists(extract_to) else set()
                    new_items = after_download - before_download
                    
                    st.info(f"Directory contents after download: {list(after_download) if after_download else 'empty'}")
                    st.info(f"New items added: {list(new_items) if new_items else 'none'}")
                    
                    # Analyze download result
                    st.info(f"Download result type: {type(download_result)}")
                    st.info(f"Download result value: {download_result}")
                    
                    # Return to original directory
                    os.chdir(original_cwd)
                    
                    if not new_items:
                        # Try alternative download methods
                        st.warning("No new files detected, trying alternative download approaches...")
                        
                        os.chdir(extract_to)
                        
                        # Try with explicit location parameter
                        try:
                            st.info("Trying download with explicit location...")
                            alt_download = dataset.download(download_format, location=extract_to)
                            after_alt = set(os.listdir(extract_to))
                            alt_new = after_alt - before_download
                            st.info(f"Alternative download result: {alt_download}")
                            st.info(f"New items after alternative download: {list(alt_new) if alt_new else 'none'}")
                            new_items = alt_new
                        except Exception as alt_error:
                            st.warning(f"Alternative download failed: {alt_error}")
                        
                        os.chdir(original_cwd)
                    
                except Exception as exec_error:
                    st.error(f"Error during Roboflow execution: {exec_error}")
                    st.error("Common issues and solutions:")
                    st.error("1. **Invalid API key**: Check your Roboflow API key")
                    st.error("2. **Wrong project/workspace name**: Verify project name and workspace")
                    st.error("3. **Network issues**: Check internet connection")
                    st.error("4. **Permission issues**: Ensure project is public or you have access")
                    raise exec_error
                
                # Step 5: Process the downloaded dataset
                status_text.text("Processing downloaded dataset...")
                progress_bar.progress(0.7)
                
                # Handle different return types from Roboflow download
                dataset_folder = None
                download_path = None
                
                # Extract path from download result
                if isinstance(download_result, str):
                    download_path = download_result
                elif hasattr(download_result, 'location'):
                    download_path = download_result.location
                elif hasattr(download_result, 'path'):
                    download_path = download_result.path
                else:
                    # Fallback: download_result might be a dataset object
                    download_path = None
                
                st.info(f"Extracted download path: {download_path}")
                
                # Check if download_path is directly usable
                if download_path and isinstance(download_path, str) and os.path.exists(download_path):
                    if os.path.isdir(download_path):
                        dataset_folder = download_path
                    else:
                        # If it's a file, use its parent directory
                        dataset_folder = os.path.dirname(download_path)
                
                # If not found, search in extract_to directory
                if not dataset_folder or not os.path.exists(dataset_folder):
                    # List all items in extract_to directory
                    try:
                        all_items = os.listdir(extract_to)
                        downloaded_folders = [d for d in all_items 
                                            if os.path.isdir(os.path.join(extract_to, d))]
                        
                        if downloaded_folders:
                            # Use the most recent folder (or first if timestamps aren't available)
                            dataset_folder = os.path.join(extract_to, downloaded_folders[-1])
                            st.info(f"Found dataset subfolder: `{downloaded_folders[-1]}`")
                        else:
                            # No subdirectories found, check if images are directly in extract_to
                            files_in_extract = [f for f in all_items 
                                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            if files_in_extract:
                                dataset_folder = extract_to
                                st.info(f"Images found directly in extract folder: {len(files_in_extract)} files")
                            
                    except Exception as e:
                        st.warning(f"Error listing directory {extract_to}: {e}")
                
                # If still not found, use extract_to as fallback
                if not dataset_folder or not os.path.exists(dataset_folder):
                    # Final check for any images in the directory tree
                    image_files = []
                    for root, dirs, files in os.walk(extract_to):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                image_files.append(os.path.join(root, file))
                    
                    if image_files:
                        # Find the common parent directory of all images
                        if len(image_files) > 1:
                            common_path = os.path.commonpath([os.path.dirname(f) for f in image_files])
                            dataset_folder = common_path
                        else:
                            dataset_folder = os.path.dirname(image_files[0])
                        st.info(f"Found {len(image_files)} images, using common directory: `{dataset_folder}`")
                    else:
                        # List what we actually found for debugging
                        dir_contents = []
                        try:
                            for root, dirs, files in os.walk(extract_to):
                                dir_contents.append(f"Directory: {root}")
                                for d in dirs[:5]:  # Limit to first 5 for readability
                                    dir_contents.append(f"  Subfolder: {d}")
                                for f in files[:10]:  # Limit to first 10 files
                                    dir_contents.append(f"  File: {f}")
                        except Exception as e:
                            dir_contents = [f"Error reading directory: {e}"]
                        
                        # Show troubleshooting suggestions
                        st.error("❌ No images found after Roboflow download!")
                        st.error("**Troubleshooting steps:**")
                        st.error("1. **Check API Key**: Ensure your Roboflow API key is valid")
                        st.error("2. **Verify Project Access**: Make sure the project exists and is accessible")
                        st.error("3. **Try Different Format**: Try 'yolov8' instead of 'folder' format")
                        st.error("4. **Check Project Version**: Ensure the version number is correct")
                        st.error("5. **Internet Connection**: Verify network connectivity")
                        
                        with st.expander("Debug Information"):
                            st.text("Download result details:")
                            st.text(f"- Result type: {type(download_result)}")
                            st.text(f"- Result value: {download_result}")
                            st.text(f"- Expected directory: {extract_to}")
                            st.text("\nDirectory contents found:")
                            st.text("\n".join(dir_contents[:20]))
                        
                        return 0
                
                st.success(f"✅ Dataset found at: `{dataset_folder}`")
                
                # Step 6: Organize dataset for testing
                status_text.text("Organizing dataset for testing...")
                progress_bar.progress(0.9)
                
                processed_count = organize_dataset_for_testing(dataset_folder, dataset_type, max_images, use_test_split)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Dataset import completed!")
                
                # Store dataset info in session state
                if 'imported_datasets' not in st.session_state:
                    st.session_state.imported_datasets = []
                
                dataset_info = {
                    'name': os.path.basename(dataset_folder),
                    'path': dataset_folder,
                    'type': dataset_type,
                    'image_count': processed_count,
                    'imported_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.imported_datasets.append(dataset_info)
                
                st.success(f"✅ Successfully imported {processed_count} images from Roboflow dataset: `{os.path.basename(dataset_folder)}`")
                
    except Exception as e:
        st.error(f"❌ Failed to import dataset: {str(e)}")
        st.info("**Next steps:**")
        st.info("1. Double-check your Roboflow code snippet")
        st.info("2. Verify your API key has the correct permissions")
        st.info("3. Ensure the project and version exist")
        st.info("4. Try a different download format (e.g., 'yolov8')")
        
        # Show the error in an expandable section for debugging
        with st.expander("Error Details (for debugging)"):
            import traceback
            st.text(traceback.format_exc())


def organize_dataset_for_testing(dataset_folder: str, dataset_type: str, max_images: int, use_test_split: bool) -> int:
    """Organize the downloaded dataset for testing purposes."""
    
    # Find test images with more flexible search
    test_images = []
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
    
    # More comprehensive test folder patterns
    test_patterns = [
        os.path.join(dataset_folder, "test"),
        os.path.join(dataset_folder, "valid"),
        os.path.join(dataset_folder, "validation"),
        os.path.join(dataset_folder, "test", "images"),
        os.path.join(dataset_folder, "valid", "images"),
        os.path.join(dataset_folder, "validation", "images"),
        os.path.join(dataset_folder, "images"),
        os.path.join(dataset_folder, "train"),
        os.path.join(dataset_folder, "train", "images"),
        dataset_folder  # fallback to main folder
    ]
    
    # Debug: Show what directories exist
    st.info(f"Searching for images in dataset folder: `{dataset_folder}`")
    
    # Try each pattern to find images
    for pattern_dir in test_patterns:
        if os.path.exists(pattern_dir):
            found_in_this_dir = []
            
            # Search for images directly in this directory
            for ext in image_extensions:
                direct_images = glob.glob(os.path.join(pattern_dir, f"*{ext}"))
                found_in_this_dir.extend(direct_images)
            
            # Also search recursively in subdirectories
            for ext in image_extensions:
                recursive_images = glob.glob(os.path.join(pattern_dir, "**", f"*{ext}"), recursive=True)
                found_in_this_dir.extend(recursive_images)
            
            if found_in_this_dir:
                st.success(f"Found {len(found_in_this_dir)} images in `{os.path.basename(pattern_dir) or 'root'}`")
                test_images.extend(found_in_this_dir)
            else:
                # Show what's actually in this directory
                try:
                    contents = os.listdir(pattern_dir)
                    if contents:
                        st.info(f"No images in `{os.path.basename(pattern_dir)}` - Found: {contents[:5]}" + (f" ... and {len(contents)-5} more" if len(contents) > 5 else ""))
                    else:
                        st.info(f"`{os.path.basename(pattern_dir)}` is empty")
                except Exception as e:
                    st.warning(f"Cannot read directory `{os.path.basename(pattern_dir)}`: {e}")
            
            # If we found images and user wants test split only, stop here
            if test_images and use_test_split and pattern_dir != dataset_folder:
                break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_images = []
    for img in test_images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)
    test_images = unique_images
    
    # If still no images found, do a comprehensive search
    if not test_images:
        st.info("No images found in standard folders, performing comprehensive search...")
        
        # Walk through ALL subdirectories
        all_found_images = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    all_found_images.append(full_path)
                    
        if all_found_images:
            st.success(f"Comprehensive search found {len(all_found_images)} images across all subdirectories!")
            test_images = all_found_images
        else:
            # Show detailed directory structure for debugging
            st.error("No images found anywhere in the dataset!")
            st.subheader("Complete Directory Structure Debug")
            
            directory_info = []
            file_types_found = set()
            
            try:
                for root, dirs, files in os.walk(dataset_folder):
                    level = root.replace(dataset_folder, '').count(os.sep)
                    indent = '  ' * level
                    directory_info.append(f"{indent}📁 {os.path.basename(root)}/")
                    
                    # Show files with their extensions
                    for file in files[:10]:  # Limit to first 10 files
                        file_ext = os.path.splitext(file)[1].lower()
                        file_types_found.add(file_ext)
                        directory_info.append(f"{indent}  📄 {file}")
                    
                    if len(files) > 10:
                        directory_info.append(f"{indent}  ... and {len(files) - 10} more files")
                
                # Show directory structure
                st.text("\n".join(directory_info[:50]))  # Limit output
                
                # Show file types found
                if file_types_found:
                    st.info(f"File types found: {', '.join(sorted(file_types_found))}")
                    st.info(f"Looking for: {', '.join(image_extensions)}")
                    
            except Exception as e:
                st.error(f"Error reading directory structure: {e}")
            
            return 0
    
    # Limit number of images
    if len(test_images) > max_images:
        st.warning(f"Found {len(test_images)} images, limiting to {max_images} as requested")
        test_images = test_images[:max_images]
    
    # Create a test_images folder in the dataset directory
    test_folder = os.path.join(dataset_folder, "test_images_for_pipeline")
    os.makedirs(test_folder, exist_ok=True)
    
    # Copy selected images to test folder
    copied_count = 0
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            dest_path = os.path.join(test_folder, f"test_{i:04d}_{os.path.basename(img_path)}")
            try:
                shutil.copy2(img_path, dest_path)
                copied_count += 1
            except Exception as e:
                st.warning(f"Failed to copy {os.path.basename(img_path)}: {str(e)}")
    
    st.success(f"Successfully organized {copied_count} images for testing!")
    return copied_count


def count_images_in_dataset(dataset_path: str) -> int:
    """Count the number of images in a dataset folder."""
    count = 0
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        count += len(glob.glob(os.path.join(dataset_path, "**", f"*{ext}"), recursive=True))
        count += len(glob.glob(os.path.join(dataset_path, "**", f"*{ext.upper()}"), recursive=True))
    return count


def get_dataset_images(dataset_name: str) -> List[str]:
    """Get list of test images from an imported dataset."""
    dataset_path = os.path.join("roboflow_datasets", dataset_name)
    test_folder = os.path.join(dataset_path, "test_images_for_pipeline")
    
    if not os.path.exists(test_folder):
        return []
    
    images = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        images.extend(glob.glob(os.path.join(test_folder, f"*{ext}")))
        images.extend(glob.glob(os.path.join(test_folder, f"*{ext.upper()}")))
    
    return sorted(images)


# =============================================================================
# CACHED RESOURCE LOADERS
# =============================================================================

@st.cache_resource
def load_nlp_model():
    """Load IndoBERT model and tokenizer directly with caching."""
    import torch
    from transformers import AutoTokenizer, EncoderDecoderModel, GenerationConfig
    
    # Model is in a subfolder on HuggingFace
    model_repo = "ZerXXX/indobert-corrector"
    subfolder = "indoBERT-best-corrector"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_repo, 
        subfolder=subfolder,
        force_download=False
    )
    model = EncoderDecoderModel.from_pretrained(
        model_repo, 
        subfolder=subfolder,
        force_download=False
    )
    
    # Explicitly set token IDs from config.json (required for generation)
    # These values are from the model's config.json on HuggingFace
    model.config.decoder_start_token_id = 2  # [CLS] token
    model.config.eos_token_id = 3            # [SEP] token
    model.config.pad_token_id = 0            # [PAD] token
    model.config.bos_token_id = 2            # Same as decoder_start
    
    # Also set on generation_config
    if model.generation_config is not None:
        model.generation_config.decoder_start_token_id = 2
        model.generation_config.eos_token_id = 3
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 2
    
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device


class TestIndoBERTCorrector:
    """IndoBERT corrector for testing - uses cached model loader."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = 64
        self.num_beams = 4
    
    def correct(self, text: str) -> str:
        if not text or not text.strip():
            return text
        try:
            import torch
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    do_sample=False,
                    early_stopping=True,
                    decoder_start_token_id=2,
                    eos_token_id=3,
                    pad_token_id=0,
                    bos_token_id=2
                )
            
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected if corrected else text
        except Exception as e:
            st.error(f"Correction error: {e}")
            return text
    
    def get_suggestions(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        try:
            import torch
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    num_return_sequences=min(self.num_beams, 3),
                    do_sample=False,
                    early_stopping=True
                )
            
            suggestions = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                if decoded and decoded != text and decoded not in suggestions:
                    suggestions.append(decoded)
            return suggestions
        except Exception as e:
            return []


def get_nlp_corrector():
    """Get NLP corrector instance using cached model."""
    model, tokenizer, device = load_nlp_model()
    return TestIndoBERTCorrector(model, tokenizer, device)


@st.cache_resource
def load_eye_analyzer():
    """Load MediaPipe eye analyzer with caching."""
    return EyeAnalyzer()


@st.cache_resource
def load_yolo_classifier(model_path: str, use_gpu: bool):
    """Load YOLO classifier with caching."""
    return YOLOEyeClassifier(model_path, use_gpu)


# =============================================================================
# NLP CORRECTION TEST PAGE
# =============================================================================

def render_nlp_test():
    """Render NLP text correction test page."""
    st.header("NLP Text Correction Test")
    st.markdown("**Model:** IndoBERT Seq2Seq (`ZerXXX/indobert-corrector`)")
    
    # Initialize session state
    if 'nlp_history' not in st.session_state:
        st.session_state.nlp_history = []
    
    # Load model
    with st.spinner("Loading IndoBERT model..."):
        try:
            corrector = get_nlp_corrector()
            st.success(f"Model loaded | Device: `{corrector.device}` | Max length: {corrector.max_length} | Beams: {corrector.num_beams}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # Predefined test cases
        st.markdown("**Quick Test Cases:**")
        test_cases = ["slmt pagi", "trm ksh", "ap kbr", "sya mau mkn", "hlo dunia"]
        
        test_cols = st.columns(len(test_cases))
        for i, (tc_col, tc) in enumerate(zip(test_cols, test_cases)):
            with tc_col:
                if st.button(tc, key=f"tc_{i}", use_container_width=True):
                    st.session_state.nlp_input = tc
        
        # Text input
        input_text = st.text_area(
            "Enter text to correct:",
            value=st.session_state.get('nlp_input', ''),
            height=100,
            key="nlp_text_input"
        )
        
        # Process button
        if st.button("Correct Text", type="primary", use_container_width=True):
            if input_text.strip():
                # Run correction
                start_time = time.time()
                corrected = corrector.correct(input_text)
                inference_time = (time.time() - start_time) * 1000
                
                # Get suggestions
                suggestions = corrector.get_suggestions(input_text)
                
                # Store result
                result = {
                    'input': input_text,
                    'output': corrected,
                    'changed': input_text != corrected,
                    'suggestions': suggestions,
                    'time_ms': inference_time,
                    'timestamp': time.strftime("%H:%M:%S")
                }
                st.session_state.nlp_history.insert(0, result)
                st.session_state.nlp_last_result = result
    
    with col2:
        st.subheader("Results")
        
        if 'nlp_last_result' in st.session_state:
            result = st.session_state.nlp_last_result
            
            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Inference Time", f"{result['time_ms']:.1f} ms")
            with m2:
                st.metric("Changed", "Yes" if result['changed'] else "No")
            with m3:
                st.metric("Suggestions", len(result['suggestions']))
            
            # Output
            st.markdown("**Corrected Output:**")
            st.code(result['output'], language=None)
            
            # Comparison
            if result['changed']:
                st.markdown("**Comparison:**")
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.markdown(f"Input: `{result['input']}`")
                with comp_col2:
                    st.markdown(f"Output: `{result['output']}`")
            
            # Suggestions
            if result['suggestions']:
                st.markdown("**Suggestions:**")
                for i, sug in enumerate(result['suggestions'], 1):
                    st.markdown(f"{i}. `{sug}`")
    
    # History section
    st.divider()
    st.subheader("Correction History")
    
    if st.session_state.nlp_history:
        # Summary stats
        total_time = sum(r['time_ms'] for r in st.session_state.nlp_history)
        avg_time = total_time / len(st.session_state.nlp_history)
        changed_count = sum(1 for r in st.session_state.nlp_history if r['changed'])
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Total Corrections", len(st.session_state.nlp_history))
        with stat_cols[1]:
            st.metric("Changed", f"{changed_count} ({100*changed_count/len(st.session_state.nlp_history):.0f}%)")
        with stat_cols[2]:
            st.metric("Total Time", f"{total_time:.1f} ms")
        with stat_cols[3]:
            st.metric("Avg Time", f"{avg_time:.1f} ms")
        
        # History table
        df = pd.DataFrame([
            {
                'Time': r['timestamp'],
                'Input': r['input'],
                'Output': r['output'],
                'Changed': 'Yes' if r['changed'] else 'No',
                'Latency (ms)': f"{r['time_ms']:.1f}"
            }
            for r in st.session_state.nlp_history[:20]
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if st.button("Clear History"):
            st.session_state.nlp_history = []
            if 'nlp_last_result' in st.session_state:
                del st.session_state.nlp_last_result
            st.rerun()
    else:
        st.info("No corrections yet. Enter text above to start testing.")


# =============================================================================
# YOLO CLASSIFIER TEST PAGE
# =============================================================================

def render_yolo_test():
    """Render YOLO eye classification test page."""
    st.header("YOLO Eye State Classification Test")
    
    # Roboflow import section
    render_roboflow_import_section()
    
    st.divider()
    
    # Initialize session state
    if 'yolo_running' not in st.session_state:
        st.session_state.yolo_running = False
    if 'yolo_stats' not in st.session_state:
        st.session_state.yolo_stats = {
            'frames': 0,
            'open': 0,
            'closed': 0,
            'total_time': 0,
            'confidences': []
        }
    if 'batch_test_results' not in st.session_state:
        st.session_state.batch_test_results = []
    
    # Config
    config = SystemConfig()
    
    # Sidebar config
    with st.sidebar:
        st.subheader("YOLO Settings")
        use_gpu = st.checkbox("Use GPU", value=True)
        
        # Dataset selection
        st.subheader("Batch Testing")
        available_datasets = []
        if os.path.exists("roboflow_datasets"):
            available_datasets = [d for d in os.listdir("roboflow_datasets") 
                                 if os.path.isdir(os.path.join("roboflow_datasets", d))]
        
        if available_datasets:
            selected_dataset = st.selectbox("Select imported dataset:", 
                                           ["None"] + available_datasets)
            
            if selected_dataset != "None":
                dataset_images = get_dataset_images(selected_dataset)
                st.info(f"Dataset: {len(dataset_images)} images")
                
                if st.button("Run Batch Test", use_container_width=True):
                    run_yolo_batch_test(selected_dataset, config, use_gpu)
        else:
            st.info("No imported datasets. Use the import section above.")
    
    # Load models
    col_status = st.columns(2)
    
    with col_status[0]:
        with st.spinner("Loading YOLO model..."):
            try:
                classifier = load_yolo_classifier(config.yolo_model_path, use_gpu)
                st.success(f"YOLO loaded | GPU: {classifier.use_gpu}")
            except Exception as e:
                st.error(f"YOLO load failed: {e}")
                return
    
    with col_status[1]:
        with st.spinner("Loading FaceLandmarker..."):
            try:
                eye_analyzer = load_eye_analyzer()
                st.success("FaceLandmarker loaded")
            except Exception as e:
                st.error(f"FaceLandmarker load failed: {e}")
                return
    
    # Test mode selection
    test_mode = st.radio("Test Mode:", 
                        ["Real-time (Webcam)", "Batch Test (Imported Dataset)"], 
                        horizontal=True)
    
    if test_mode == "Real-time (Webcam)":
        render_realtime_yolo_test(classifier, eye_analyzer, config)
    else:
        render_batch_yolo_test()


def render_realtime_yolo_test(classifier, eye_analyzer, config):
    """Render real-time YOLO test with webcam."""
    # Controls
    ctrl_cols = st.columns([1, 1, 1])
    with ctrl_cols[0]:
        start_btn = st.button("Start", use_container_width=True, type="primary")
    with ctrl_cols[1]:
        stop_btn = st.button("Stop", use_container_width=True)
    with ctrl_cols[2]:
        reset_btn = st.button("Reset Stats", use_container_width=True)
    
    if start_btn:
        st.session_state.yolo_running = True
    if stop_btn:
        st.session_state.yolo_running = False
    if reset_btn:
        st.session_state.yolo_stats = {
            'frames': 0, 'open': 0, 'closed': 0, 'total_time': 0, 'confidences': []
        }
    
    # Main display area
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
    
    with col_metrics:
        st.subheader("Real-time Metrics")
        state_display = st.empty()
        confidence_display = st.empty()
        probs_display = st.empty()
        timing_display = st.empty()
        
        st.subheader("Session Stats")
        stats_display = st.empty()
    
    # Eye crops display
    st.subheader("Eye Crops")
    crop_cols = st.columns(2)
    left_crop_display = crop_cols[0].empty()
    right_crop_display = crop_cols[1].empty()
    
    # Video loop
    if st.session_state.yolo_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("Cannot open webcam")
            st.session_state.yolo_running = False
        else:
            try:
                while st.session_state.yolo_running:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    eye_data, annotated = eye_analyzer.process_frame(frame, config)
                    
                    # YOLO inference
                    start_time = time.time()
                    yolo_result = classifier.classify_dual_eye(
                        eye_data.left_crop, eye_data.right_crop
                    )
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Update stats
                    stats = st.session_state.yolo_stats
                    stats['frames'] += 1
                    stats['total_time'] += inference_time
                    if yolo_result.state == EyeState.OPEN:
                        stats['open'] += 1
                    elif yolo_result.state == EyeState.CLOSED:
                        stats['closed'] += 1
                    stats['confidences'].append(yolo_result.confidence)
                    if len(stats['confidences']) > 100:
                        stats['confidences'] = stats['confidences'][-100:]
                    
                    # Draw on frame
                    state_color = (0, 255, 0) if yolo_result.state == EyeState.OPEN else (0, 0, 255)
                    cv2.putText(annotated, f"{yolo_result.state.value.upper()}", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
                    cv2.putText(annotated, f"Conf: {yolo_result.confidence:.1%}", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Display frame
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Update metrics
                    state_emoji = "Open" if yolo_result.state == EyeState.OPEN else "Closed"
                    state_display.metric("Eye State", f"{state_emoji}")
                    confidence_display.metric("Confidence", f"{yolo_result.confidence:.1%}")
                    probs_display.markdown(f"""
                    **Probabilities:**
                    - Open: `{yolo_result.open_prob:.2%}`
                    - Closed: `{yolo_result.closed_prob:.2%}`
                    """)
                    timing_display.metric("Inference", f"{inference_time:.1f} ms")
                    
                    # Session stats
                    avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
                    stats_display.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Frames | {stats['frames']} |
                    | Open | {stats['open']} ({100*stats['open']/max(stats['frames'],1):.1f}%) |
                    | Closed | {stats['closed']} ({100*stats['closed']/max(stats['frames'],1):.1f}%) |
                    | Avg Conf | {avg_conf:.1%} |
                    | Avg Time | {stats['total_time']/max(stats['frames'],1):.1f} ms |
                    """)
                    
                    # Eye crops
                    if eye_data.left_crop is not None and eye_data.left_crop.size > 0:
                        left_crop_display.image(
                            cv2.cvtColor(eye_data.left_crop, cv2.COLOR_BGR2RGB),
                            caption="Left Eye", use_container_width=True
                        )
                    if eye_data.right_crop is not None and eye_data.right_crop.size > 0:
                        right_crop_display.image(
                            cv2.cvtColor(eye_data.right_crop, cv2.COLOR_BGR2RGB),
                            caption="Right Eye", use_container_width=True
                        )
                    
            finally:
                cap.release()
    else:
        video_placeholder.info("Click 'Start' to begin YOLO classification test")
        
        # Show final stats if available
        stats = st.session_state.yolo_stats
        if stats['frames'] > 0:
            st.divider()
            st.subheader("Final Session Metrics")
            
            m_cols = st.columns(5)
            m_cols[0].metric("Total Frames", stats['frames'])
            m_cols[1].metric("Open", f"{stats['open']} ({100*stats['open']/stats['frames']:.1f}%)")
            m_cols[2].metric("Closed", f"{stats['closed']} ({100*stats['closed']/stats['frames']:.1f}%)")
            m_cols[3].metric("Avg Confidence", f"{np.mean(stats['confidences']):.1%}" if stats['confidences'] else "N/A")
            m_cols[4].metric("Avg Inference", f"{stats['total_time']/stats['frames']:.1f} ms")


def render_batch_yolo_test():
    """Render batch testing results with comprehensive evaluation metrics."""
    st.subheader("Batch Test Results & Evaluation Metrics")
    
    if 'batch_test_results' in st.session_state and st.session_state.batch_test_results:
        # Show latest results
        latest_result = st.session_state.batch_test_results[-1]
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset", latest_result['dataset'])
        col2.metric("Total Images", latest_result['total_images'])
        col3.metric("Avg Confidence", f"{latest_result['avg_confidence']:.1%}")
        col4.metric("Avg Inference Time", f"{latest_result['avg_time']:.1f} ms")
        
        # Evaluation metrics
        if 'detailed_results' in latest_result:
            df = pd.DataFrame(latest_result['detailed_results'])
            
            # Distribution analysis
            st.subheader("📊 Evaluation Metrics")
            
            # State distribution
            state_counts = df['state'].value_counts()
            metric_cols = st.columns(len(state_counts) + 2)
            
            for i, (state, count) in enumerate(state_counts.items()):
                metric_cols[i].metric(f"{state.title()} Eyes", f"{count} ({100*count/len(df):.1f}%)")
            
            # Confidence statistics
            metric_cols[-2].metric("High Confidence (>80%)", f"{len(df[df['confidence'] > 0.8])} ({100*len(df[df['confidence'] > 0.8])/len(df):.1f}%)")
            metric_cols[-1].metric("Low Confidence (<50%)", f"{len(df[df['confidence'] < 0.5])} ({100*len(df[df['confidence'] < 0.5])/len(df):.1f}%)")
            
            # Confidence distribution chart
            st.subheader("📈 Confidence Distribution")
            conf_col1, conf_col2 = st.columns(2)
            
            with conf_col1:
                st.markdown("**Confidence Histogram**")
                fig_hist = px.histogram(df, x='confidence', nbins=20, 
                                       title="Confidence Score Distribution",
                                       labels={'confidence': 'Confidence Score', 'count': 'Number of Images'})
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with conf_col2:
                st.markdown("**State vs Confidence**")
                fig_box = px.box(df, x='state', y='confidence', 
                               title="Confidence Distribution by Eye State",
                               labels={'state': 'Eye State', 'confidence': 'Confidence Score'})
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Performance analysis
            st.subheader("⚡ Performance Analysis")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("Min Inference Time", f"{df['inference_time_ms'].min():.1f} ms")
                st.metric("Max Inference Time", f"{df['inference_time_ms'].max():.1f} ms")
                st.metric("Std Inference Time", f"{df['inference_time_ms'].std():.1f} ms")
            
            with perf_col2:
                st.metric("Min Confidence", f"{df['confidence'].min():.1%}")
                st.metric("Max Confidence", f"{df['confidence'].max():.1%}")
                st.metric("Std Confidence", f"{df['confidence'].std():.3f}")
            
            with perf_col3:
                # Time vs confidence correlation
                corr = df['inference_time_ms'].corr(df['confidence'])
                st.metric("Time-Confidence Correlation", f"{corr:.3f}")
                
                # Probability spread analysis
                df['prob_spread'] = abs(df['open_prob'] - df['closed_prob'])
                st.metric("Avg Probability Spread", f"{df['prob_spread'].mean():.3f}")
                st.metric("Low Spread (<0.3)", f"{len(df[df['prob_spread'] < 0.3])} images")
            
            # Detailed results table
            st.subheader("🔍 Detailed Per-Image Results")
            
            # Add result analysis columns
            df_display = df.copy()
            df_display['confidence_level'] = pd.cut(df_display['confidence'], 
                                                   bins=[0, 0.5, 0.8, 1.0], 
                                                   labels=['Low', 'Medium', 'High'])
            df_display = df_display.round({'confidence': 3, 'open_prob': 3, 'closed_prob': 3, 'inference_time_ms': 1, 'prob_spread': 3})
            
            # Filter options
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                state_filter = st.selectbox("Filter by State:", ["All"] + list(df['state'].unique()))
            with filter_col2:
                conf_filter = st.selectbox("Filter by Confidence:", ["All", "High (>80%)", "Medium (50-80%)", "Low (<50%)"])
            with filter_col3:
                sort_by = st.selectbox("Sort by:", ["confidence", "inference_time_ms", "prob_spread", "image"])
            
            # Apply filters
            df_filtered = df_display.copy()
            if state_filter != "All":
                df_filtered = df_filtered[df_filtered['state'] == state_filter]
            if conf_filter != "All":
                if conf_filter == "High (>80%)":
                    df_filtered = df_filtered[df_filtered['confidence'] > 0.8]
                elif conf_filter == "Medium (50-80%)":
                    df_filtered = df_filtered[(df_filtered['confidence'] >= 0.5) & (df_filtered['confidence'] <= 0.8)]
                elif conf_filter == "Low (<50%)":
                    df_filtered = df_filtered[df_filtered['confidence'] < 0.5]
            
            df_filtered = df_filtered.sort_values(by=sort_by, ascending=False)
            
            # Show filtered count
            st.info(f"Showing {len(df_filtered)} of {len(df)} images")
            
            # Style the dataframe
            def highlight_confidence(val):
                if val > 0.8:
                    return 'background-color: #d4edda; color: #155724'  # Green
                elif val < 0.5:
                    return 'background-color: #f8d7da; color: #721c24'  # Red
                else:
                    return 'background-color: #fff3cd; color: #856404'  # Yellow
            
            styled_df = df_filtered.style.applymap(highlight_confidence, subset=['confidence'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Summary statistics
            st.subheader("📋 Summary Statistics")
            summary_stats = {
                'Total Images': len(df),
                'Mean Confidence': f"{df['confidence'].mean():.3f}",
                'Median Confidence': f"{df['confidence'].median():.3f}",
                'Mean Inference Time': f"{df['inference_time_ms'].mean():.1f} ms",
                'Images with High Confidence': f"{len(df[df['confidence'] > 0.8])} ({100*len(df[df['confidence'] > 0.8])/len(df):.1f}%)",
                'Images with Uncertain Predictions': f"{len(df[df['prob_spread'] < 0.3])} ({100*len(df[df['prob_spread'] < 0.3])/len(df):.1f}%)"
            }
            
            stats_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
            st.table(stats_df)
        
        # Download results
        if st.button("💾 Download Detailed Results as CSV", type="primary"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"yolo_evaluation_{latest_result['dataset']}_{int(time.time())}.csv",
                mime='text/csv'
            )
    else:
        st.info("No batch test results available. Run a batch test from the sidebar.")


def run_yolo_batch_test(dataset_name: str, config, use_gpu: bool):
    """Run batch test on imported dataset with comprehensive metrics."""
    try:
        # Load models
        classifier = load_yolo_classifier(config.yolo_model_path, use_gpu)
        eye_analyzer = load_eye_analyzer()
        
        # Get dataset images
        dataset_images = get_dataset_images(dataset_name)
        
        if not dataset_images:
            st.error(f"No images found in dataset: {dataset_name}")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_time = 0
        inference_times = []
        confidences = []
        
        for i, img_path in enumerate(dataset_images):
            status_text.text(f"Processing image {i+1}/{len(dataset_images)}: {os.path.basename(img_path)}")
            progress_bar.progress((i + 1) / len(dataset_images))
            
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Process with eye analyzer
            try:
                eye_data, _ = eye_analyzer.process_frame(image, config)
                
                # YOLO classification
                start_time = time.time()
                yolo_result = classifier.classify_dual_eye(eye_data.left_crop, eye_data.right_crop)
                inference_time = (time.time() - start_time) * 1000
                total_time += inference_time
                
                # Store detailed result
                result = {
                    'image': os.path.basename(img_path),
                    'state': yolo_result.state.value,
                    'confidence': yolo_result.confidence,
                    'open_prob': yolo_result.open_prob,
                    'closed_prob': yolo_result.closed_prob,
                    'inference_time_ms': inference_time,
                    'eye_crops_available': eye_data.left_crop is not None and eye_data.right_crop is not None,
                    'landmarks_detected': eye_data.landmarks_detected,
                    'ear_value': eye_data.avg_ear if eye_data.landmarks_detected else None,
                }
                
                results.append(result)
                inference_times.append(inference_time)
                confidences.append(yolo_result.confidence)
                
            except Exception as e:
                st.warning(f"Error processing {os.path.basename(img_path)}: {str(e)}")
                # Add error result
                results.append({
                    'image': os.path.basename(img_path),
                    'state': 'error',
                    'confidence': 0.0,
                    'open_prob': 0.0,
                    'closed_prob': 0.0,
                    'inference_time_ms': 0.0,
                    'eye_crops_available': False,
                    'landmarks_detected': False,
                    'ear_value': None,
                    'error': str(e)
                })
        
        # Calculate comprehensive statistics
        valid_results = [r for r in results if r['state'] != 'error']
        avg_confidence = np.mean([r['confidence'] for r in valid_results]) if valid_results else 0
        avg_time = total_time / len(valid_results) if valid_results else 0
        
        # Additional metrics
        state_distribution = {}
        for result in valid_results:
            state = result['state']
            state_distribution[state] = state_distribution.get(state, 0) + 1
        
        # Store comprehensive results
        batch_result = {
            'dataset': dataset_name,
            'total_images': len(results),
            'valid_images': len(valid_results),
            'error_images': len(results) - len(valid_results),
            'avg_confidence': avg_confidence,
            'avg_time': avg_time,
            'min_confidence': min([r['confidence'] for r in valid_results]) if valid_results else 0,
            'max_confidence': max([r['confidence'] for r in valid_results]) if valid_results else 0,
            'confidence_std': np.std([r['confidence'] for r in valid_results]) if valid_results else 0,
            'state_distribution': state_distribution,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'detailed_results': results
        }
        
        st.session_state.batch_test_results.append(batch_result)
        
        status_text.text("✅ Batch test completed!")
        progress_bar.progress(1.0)
        
        # Show quick summary
        col1, col2, col3, col4 = st.columns(4)
        col1.success(f"✅ Processed: {len(valid_results)}")
        col2.info(f"⚡ Avg Time: {avg_time:.1f}ms") 
        col3.info(f"🎯 Avg Confidence: {avg_confidence:.1%}")
        col4.warning(f"❌ Errors: {len(results) - len(valid_results)}" if len(results) - len(valid_results) > 0 else f"✅ No Errors")
        
        st.success(f"Completed comprehensive batch test on {len(results)} images!")
        
    except Exception as e:
        st.error(f"Batch test failed: {str(e)}")


# =============================================================================
# EAR ANALYSIS TEST PAGE
# =============================================================================

def render_ear_test():
    """Render EAR analysis test page."""
    st.header("Eye Aspect Ratio (EAR) Analysis Test")
    
    # Roboflow import section
    render_roboflow_import_section()
    
    st.divider()
    
    # Initialize session state
    if 'ear_running' not in st.session_state:
        st.session_state.ear_running = False
    if 'ear_stats' not in st.session_state:
        st.session_state.ear_stats = {
            'frames': 0,
            'detected': 0,
            'ear_values': [],
            'left_ear_values': [],
            'right_ear_values': []
        }
    if 'ear_calibration' not in st.session_state:
        st.session_state.ear_calibration = {'open': None, 'closed': None}
    if 'ear_batch_results' not in st.session_state:
        st.session_state.ear_batch_results = []
    
    config = SystemConfig()
    
    # Sidebar config
    with st.sidebar:
        st.subheader("EAR Settings")
        ear_min = st.slider("EAR Min (closed)", 0.05, 0.25, config.ear_min, 0.01)
        ear_max = st.slider("EAR Max (open)", 0.25, 0.50, config.ear_max, 0.01)
        config.ear_min = ear_min
        config.ear_max = ear_max
        
        # Dataset selection for EAR testing
        st.subheader("EAR Batch Testing")
        available_datasets = []
        if os.path.exists("roboflow_datasets"):
            available_datasets = [d for d in os.listdir("roboflow_datasets") 
                                 if os.path.isdir(os.path.join("roboflow_datasets", d))]
        
        if available_datasets:
            selected_dataset = st.selectbox("Select dataset for EAR:", 
                                           ["None"] + available_datasets)
            
            if selected_dataset != "None":
                dataset_images = get_dataset_images(selected_dataset)
                st.info(f"Dataset: {len(dataset_images)} images")
                
                if st.button("Run EAR Batch Test", use_container_width=True):
                    run_ear_batch_test(selected_dataset, config)
        else:
            st.info("No datasets. Use import section above.")
    
    # Load analyzer
    with st.spinner("Loading FaceLandmarker..."):
        try:
            eye_analyzer = load_eye_analyzer()
            st.success("FaceLandmarker loaded")
        except Exception as e:
            st.error(f"Load failed: {e}")
            return
    
    # Test mode selection
    test_mode = st.radio("EAR Test Mode:", 
                        ["Real-time (Webcam)", "Batch Analysis (Dataset)"], 
                        horizontal=True)
    
    if test_mode == "Real-time (Webcam)":
        render_realtime_ear_test(eye_analyzer, config)
    else:
        render_batch_ear_results()


def render_realtime_ear_test(eye_analyzer, config):
    """Render real-time EAR test with webcam."""
    # Controls
    ctrl_cols = st.columns([1, 1, 1, 1])
    with ctrl_cols[0]:
        if st.button("Start", use_container_width=True, type="primary"):
            st.session_state.ear_running = True
    with ctrl_cols[1]:
        if st.button("Stop", use_container_width=True):
            st.session_state.ear_running = False
    with ctrl_cols[2]:
        if st.button("Calibrate Open", use_container_width=True):
            if st.session_state.ear_stats['ear_values']:
                st.session_state.ear_calibration['open'] = np.mean(st.session_state.ear_stats['ear_values'][-30:])
    with ctrl_cols[3]:
        if st.button("Calibrate Closed", use_container_width=True):
            if st.session_state.ear_stats['ear_values']:
                st.session_state.ear_calibration['closed'] = np.mean(st.session_state.ear_stats['ear_values'][-30:])
    
    # Calibration display
    cal = st.session_state.ear_calibration
    if cal['open'] or cal['closed']:
        cal_cols = st.columns(2)
        with cal_cols[0]:
            if cal['open']:
                st.success(f"Open EAR: {cal['open']:.4f}")
        with cal_cols[1]:
            if cal['closed']:
                st.success(f"Closed EAR: {cal['closed']:.4f}")
    
    # Main display
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
        
        st.subheader("EAR History Graph")
        chart_placeholder = st.empty()
    
    with col_metrics:
        st.subheader("EAR Metrics")
        ear_display = st.empty()
        left_ear_display = st.empty()
        right_ear_display = st.empty()
        normalized_display = st.empty()
        
        st.subheader("Session Stats")
        stats_display = st.empty()
    
    # Video loop for real-time testing
    if st.session_state.ear_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("Cannot open webcam")
            st.session_state.ear_running = False
        else:
            try:
                while st.session_state.ear_running:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    eye_data, annotated = eye_analyzer.process_frame(frame, config)
                    
                    # Update stats if landmarks detected
                    stats = st.session_state.ear_stats
                    stats['frames'] += 1
                    
                    if eye_data.landmarks_detected:
                        stats['detected'] += 1
                        stats['ear_values'].append(eye_data.avg_ear)
                        stats['left_ear_values'].append(eye_data.left_ear)
                        stats['right_ear_values'].append(eye_data.right_ear)
                        
                        # Keep only last 200 values for visualization
                        for key in ['ear_values', 'left_ear_values', 'right_ear_values']:
                            if len(stats[key]) > 200:
                                stats[key] = stats[key][-200:]
                        
                        # Determine state based on normalized EAR
                        ear_state = "CLOSED" if eye_data.normalized_ear < 0.5 else "OPEN"
                        state_color = (0, 0, 255) if ear_state == "CLOSED" else (0, 255, 0)
                        
                        # Draw EAR info on frame
                        cv2.putText(annotated, f"EAR: {eye_data.avg_ear:.4f}", 
                                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(annotated, f"Norm: {eye_data.normalized_ear:.3f}", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(annotated, f"State: {ear_state}", 
                                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
                    
                    # Display frame
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Update metrics
                    if eye_data.landmarks_detected:
                        ear_display.metric("Avg EAR", f"{eye_data.avg_ear:.4f}")
                        left_ear_display.metric("Left EAR", f"{eye_data.left_ear:.4f}")
                        right_ear_display.metric("Right EAR", f"{eye_data.right_ear:.4f}")
                        normalized_display.metric("Normalized", f"{eye_data.normalized_ear:.3f}")
                        
                        # Stats
                        detection_rate = 100 * stats['detected'] / max(stats['frames'], 1)
                        avg_ear = np.mean(stats['ear_values']) if stats['ear_values'] else 0
                        ear_std = np.std(stats['ear_values']) if stats['ear_values'] else 0
                        
                        stats_display.markdown(f"""
                        | Metric | Value |
                        |--------|-------|
                        | Frames | {stats['frames']} |
                        | Detection Rate | {detection_rate:.1f}% |
                        | Avg EAR | {avg_ear:.4f} |
                        | EAR Std | {ear_std:.4f} |
                        | Data Points | {len(stats['ear_values'])} |
                        """)
                        
                        # Chart
                        if len(stats['ear_values']) > 10:
                            chart_data = pd.DataFrame({
                                'Frame': range(len(stats['ear_values'])),
                                'Average EAR': stats['ear_values'],
                                'Left EAR': stats['left_ear_values'][-len(stats['ear_values']):],
                                'Right EAR': stats['right_ear_values'][-len(stats['ear_values']):]
                            })
                            chart_placeholder.line_chart(chart_data.set_index('Frame'))
                        
            finally:
                cap.release()
    else:
        video_placeholder.info("Click 'Start' to begin EAR analysis test")
        
        # Show final stats
        stats = st.session_state.ear_stats
        if stats['frames'] > 0:
            st.divider()
            st.subheader("Final EAR Session Summary")
            
            detection_rate = 100 * stats['detected'] / stats['frames']
            avg_ear = np.mean(stats['ear_values']) if stats['ear_values'] else 0
            ear_range = (min(stats['ear_values']), max(stats['ear_values'])) if stats['ear_values'] else (0, 0)
            
            m_cols = st.columns(4)
            m_cols[0].metric("Detection Rate", f"{detection_rate:.1f}%")
            m_cols[1].metric("Avg EAR", f"{avg_ear:.4f}")
            m_cols[2].metric("EAR Range", f"{ear_range[0]:.3f} - {ear_range[1]:.3f}")
            m_cols[3].metric("Data Points", len(stats['ear_values']))


def render_batch_ear_results():
    """Render EAR batch testing results with comprehensive evaluation metrics."""
    st.subheader("EAR Batch Analysis Results & Evaluation")
    
    if 'ear_batch_results' in st.session_state and st.session_state.ear_batch_results:
        latest_result = st.session_state.ear_batch_results[-1]
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset", latest_result['dataset'])
        col2.metric("Images Processed", latest_result['processed_images'])
        col3.metric("Avg EAR", f"{latest_result['avg_ear']:.4f}")
        col4.metric("Detection Rate", f"{latest_result['detection_rate']:.1f}%")
        
        if 'detailed_results' in latest_result:
            df = pd.DataFrame(latest_result['detailed_results'])
            detected_df = df[df['landmarks_detected']].copy()
            
            # Comprehensive EAR Analysis
            st.subheader("📊 Comprehensive EAR Analysis")
            
            if len(detected_df) > 0:
                # EAR statistics with more detail
                stat_cols = st.columns(6)
                stat_cols[0].metric("Mean EAR", f"{detected_df['avg_ear'].mean():.4f}")
                stat_cols[1].metric("Median EAR", f"{detected_df['avg_ear'].median():.4f}")
                stat_cols[2].metric("Min EAR", f"{detected_df['avg_ear'].min():.4f}")
                stat_cols[3].metric("Max EAR", f"{detected_df['avg_ear'].max():.4f}")
                stat_cols[4].metric("EAR Std Dev", f"{detected_df['avg_ear'].std():.4f}")
                stat_cols[5].metric("EAR Range", f"{detected_df['avg_ear'].max() - detected_df['avg_ear'].min():.4f}")
                
                # Eye asymmetry analysis
                if 'left_ear' in detected_df.columns and 'right_ear' in detected_df.columns:
                    detected_df['ear_asymmetry'] = abs(detected_df['left_ear'] - detected_df['right_ear'])
                    
                    asym_cols = st.columns(3)
                    asym_cols[0].metric("Avg Eye Asymmetry", f"{detected_df['ear_asymmetry'].mean():.4f}")
                    asym_cols[1].metric("High Asymmetry (>0.05)", f"{len(detected_df[detected_df['ear_asymmetry'] > 0.05])} images")
                    asym_cols[2].metric("Max Asymmetry", f"{detected_df['ear_asymmetry'].max():.4f}")
                
                # EAR distribution analysis
                st.subheader("📈 EAR Distribution Analysis")
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.markdown("**EAR Distribution Histogram**")
                    fig_hist = px.histogram(detected_df, x='avg_ear', nbins=25,
                                          title="Average EAR Distribution",
                                          labels={'avg_ear': 'Average EAR', 'count': 'Number of Images'})
                    fig_hist.add_vline(x=detected_df['avg_ear'].mean(), line_dash="dash", 
                                      annotation_text=f"Mean: {detected_df['avg_ear'].mean():.3f}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with viz_col2:
                    if 'left_ear' in detected_df.columns and 'right_ear' in detected_df.columns:
                        st.markdown("**Left vs Right Eye EAR**")
                        fig_scatter = px.scatter(detected_df, x='left_ear', y='right_ear',
                                               title="Left vs Right Eye EAR Correlation",
                                               labels={'left_ear': 'Left Eye EAR', 'right_ear': 'Right Eye EAR'})
                        # Add diagonal line for perfect correlation
                        min_ear = min(detected_df['left_ear'].min(), detected_df['right_ear'].min())
                        max_ear = max(detected_df['left_ear'].max(), detected_df['right_ear'].max())
                        fig_scatter.add_trace(go.Scatter(x=[min_ear, max_ear], y=[min_ear, max_ear], 
                                                        mode='lines', name='Perfect Correlation', 
                                                        line=dict(dash='dash')))
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.markdown("**EAR Statistics Box Plot**")
                        fig_box = px.box(y=detected_df['avg_ear'], title="EAR Distribution Box Plot")
                        st.plotly_chart(fig_box, use_container_width=True)
                
                # EAR quality assessment
                st.subheader("🔍 EAR Quality Assessment")
                quality_cols = st.columns(4)
                
                # Categorize EAR values
                very_low_ear = len(detected_df[detected_df['avg_ear'] < 0.15])  # Likely closed
                low_ear = len(detected_df[(detected_df['avg_ear'] >= 0.15) & (detected_df['avg_ear'] < 0.25)])  # Partially closed
                normal_ear = len(detected_df[(detected_df['avg_ear'] >= 0.25) & (detected_df['avg_ear'] < 0.35)])  # Normal open
                high_ear = len(detected_df[detected_df['avg_ear'] >= 0.35])  # Wide open
                
                quality_cols[0].metric("Very Low EAR (<0.15)", f"{very_low_ear} ({100*very_low_ear/len(detected_df):.1f}%)")
                quality_cols[1].metric("Low EAR (0.15-0.25)", f"{low_ear} ({100*low_ear/len(detected_df):.1f}%)")
                quality_cols[2].metric("Normal EAR (0.25-0.35)", f"{normal_ear} ({100*normal_ear/len(detected_df):.1f}%)")
                quality_cols[3].metric("High EAR (>0.35)", f"{high_ear} ({100*high_ear/len(detected_df):.1f}%)")
                
            # Detection issues analysis
            failed_detection = len(df[~df['landmarks_detected']])
            if failed_detection > 0:
                st.subheader("⚠️ Detection Issues")
                st.warning(f"{failed_detection} images ({100*failed_detection/len(df):.1f}%) failed landmark detection")
            
            # Detailed results table with filtering
            st.subheader("🔍 Detailed Per-Image Results")
            
            # Filter options
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                detection_filter = st.selectbox("Filter by Detection:", ["All", "Detected Only", "Failed Detection"])
            with filter_col2:
                if len(detected_df) > 0:
                    ear_filter = st.selectbox("Filter by EAR Range:", 
                                            ["All", "Very Low (<0.15)", "Low (0.15-0.25)", "Normal (0.25-0.35)", "High (>0.35)"])
                else:
                    ear_filter = "All"
            with filter_col3:
                sort_by = st.selectbox("Sort by:", [col for col in ["avg_ear", "left_ear", "right_ear", "ear_asymmetry", "image"] 
                                     if col in df.columns])
            
            # Apply filters
            df_filtered = df.copy()
            if detection_filter == "Detected Only":
                df_filtered = df_filtered[df_filtered['landmarks_detected']]
            elif detection_filter == "Failed Detection":
                df_filtered = df_filtered[~df_filtered['landmarks_detected']]
            
            if ear_filter != "All" and len(df_filtered[df_filtered['landmarks_detected']]) > 0:
                detected_filtered = df_filtered[df_filtered['landmarks_detected']]
                if ear_filter == "Very Low (<0.15)":
                    df_filtered = detected_filtered[detected_filtered['avg_ear'] < 0.15]
                elif ear_filter == "Low (0.15-0.25)":
                    df_filtered = detected_filtered[(detected_filtered['avg_ear'] >= 0.15) & (detected_filtered['avg_ear'] < 0.25)]
                elif ear_filter == "Normal (0.25-0.35)":
                    df_filtered = detected_filtered[(detected_filtered['avg_ear'] >= 0.25) & (detected_filtered['avg_ear'] < 0.35)]
                elif ear_filter == "High (>0.35)":
                    df_filtered = detected_filtered[detected_filtered['avg_ear'] >= 0.35]
            
            if sort_by in df_filtered.columns:
                df_filtered = df_filtered.sort_values(by=sort_by, ascending=False)
            
            st.info(f"Showing {len(df_filtered)} of {len(df)} images")
            
            # Style the dataframe for better visualization
            if len(df_filtered) > 0:
                df_filtered_display = df_filtered.round(4)
                st.dataframe(df_filtered_display, use_container_width=True)
            
            # Comprehensive summary
            st.subheader("📋 Comprehensive Summary")
            if len(detected_df) > 0:
                summary_stats = {
                    'Total Images': len(df),
                    'Successfully Detected': f"{len(detected_df)} ({100*len(detected_df)/len(df):.1f}%)",
                    'Failed Detection': f"{len(df) - len(detected_df)} ({100*(len(df) - len(detected_df))/len(df):.1f}%)",
                    'Mean EAR': f"{detected_df['avg_ear'].mean():.4f}",
                    'EAR Standard Deviation': f"{detected_df['avg_ear'].std():.4f}",
                    'EAR Range': f"{detected_df['avg_ear'].min():.4f} - {detected_df['avg_ear'].max():.4f}",
                    'Likely Closed Eyes (<0.20)': f"{len(detected_df[detected_df['avg_ear'] < 0.20])} ({100*len(detected_df[detected_df['avg_ear'] < 0.20])/len(detected_df):.1f}%)",
                    'Likely Open Eyes (>0.25)': f"{len(detected_df[detected_df['avg_ear'] > 0.25])} ({100*len(detected_df[detected_df['avg_ear'] > 0.25])/len(detected_df):.1f}%)"
                }
                
                if 'ear_asymmetry' in detected_df.columns:
                    summary_stats.update({
                        'Mean Eye Asymmetry': f"{detected_df['ear_asymmetry'].mean():.4f}",
                        'High Asymmetry Cases': f"{len(detected_df[detected_df['ear_asymmetry'] > 0.05])} images"
                    })
                
                stats_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                st.table(stats_df)
            else:
                st.warning("No landmarks detected in any images. Check your dataset quality.")
        
        # Download option
        if st.button("💾 Download Detailed EAR Results as CSV", type="primary"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ear_evaluation_{latest_result['dataset']}_{int(time.time())}.csv",
                mime='text/csv'
            )
    else:
        st.info("No EAR batch test results available. Run a batch test from the sidebar.")


def run_ear_batch_test(dataset_name: str, config):
    """Run batch EAR analysis on imported dataset with comprehensive metrics."""
    try:
        eye_analyzer = load_eye_analyzer()
        dataset_images = get_dataset_images(dataset_name)
        
        if not dataset_images:
            st.error(f"No images found in dataset: {dataset_name}")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        ear_values = []
        left_ear_values = []
        right_ear_values = []
        
        for i, img_path in enumerate(dataset_images):
            status_text.text(f"Analyzing image {i+1}/{len(dataset_images)}: {os.path.basename(img_path)}")
            progress_bar.progress((i + 1) / len(dataset_images))
            
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            try:
                # Analyze with eye analyzer
                eye_data, _ = eye_analyzer.process_frame(image, config)
                
                # Store comprehensive result
                result = {
                    'image': os.path.basename(img_path),
                    'landmarks_detected': eye_data.landmarks_detected,
                    'left_ear': eye_data.left_ear if eye_data.landmarks_detected else None,
                    'right_ear': eye_data.right_ear if eye_data.landmarks_detected else None,
                    'avg_ear': eye_data.avg_ear if eye_data.landmarks_detected else None,
                    'normalized_ear': eye_data.normalized_ear if eye_data.landmarks_detected else None,
                    'image_dimensions': f"{image.shape[1]}x{image.shape[0]}" if image is not None else None,
                }
                
                results.append(result)
                
                if eye_data.landmarks_detected:
                    ear_values.append(eye_data.avg_ear)
                    left_ear_values.append(eye_data.left_ear)
                    right_ear_values.append(eye_data.right_ear)
                    
            except Exception as e:
                st.warning(f"Error analyzing {os.path.basename(img_path)}: {str(e)}")
                # Add error result
                results.append({
                    'image': os.path.basename(img_path),
                    'landmarks_detected': False,
                    'left_ear': None,
                    'right_ear': None,
                    'avg_ear': None,
                    'normalized_ear': None,
                    'image_dimensions': None,
                    'error': str(e)
                })
        
        # Calculate comprehensive statistics
        processed_images = len(results)
        detected_images = sum(1 for r in results if r['landmarks_detected'])
        detection_rate = 100 * detected_images / processed_images if processed_images > 0 else 0
        
        # EAR statistics
        if ear_values:
            avg_ear = np.mean(ear_values)
            min_ear = min(ear_values)
            max_ear = max(ear_values)
            ear_std = np.std(ear_values)
            median_ear = np.median(ear_values)
            
            # Eye asymmetry analysis
            if left_ear_values and right_ear_values:
                asymmetries = [abs(l - r) for l, r in zip(left_ear_values, right_ear_values)]
                avg_asymmetry = np.mean(asymmetries)
                max_asymmetry = max(asymmetries)
                high_asymmetry_count = sum(1 for a in asymmetries if a > 0.05)
            else:
                avg_asymmetry = max_asymmetry = high_asymmetry_count = 0
            
        else:
            avg_ear = min_ear = max_ear = ear_std = median_ear = 0
            avg_asymmetry = max_asymmetry = high_asymmetry_count = 0
        
        # Quality assessment
        quality_stats = {}
        if ear_values:
            quality_stats = {
                'very_low_ear': sum(1 for ear in ear_values if ear < 0.15),
                'low_ear': sum(1 for ear in ear_values if 0.15 <= ear < 0.25),
                'normal_ear': sum(1 for ear in ear_values if 0.25 <= ear < 0.35),
                'high_ear': sum(1 for ear in ear_values if ear >= 0.35)
            }
        
        # Store comprehensive batch result
        batch_result = {
            'dataset': dataset_name,
            'processed_images': processed_images,
            'detected_images': detected_images,
            'failed_images': processed_images - detected_images,
            'detection_rate': detection_rate,
            'avg_ear': avg_ear,
            'median_ear': median_ear,
            'min_ear': min_ear,
            'max_ear': max_ear,
            'ear_std': ear_std,
            'ear_range': max_ear - min_ear if ear_values else 0,
            'avg_asymmetry': avg_asymmetry,
            'max_asymmetry': max_asymmetry,
            'high_asymmetry_count': high_asymmetry_count,
            'quality_distribution': quality_stats,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'detailed_results': results
        }
        
        st.session_state.ear_batch_results.append(batch_result)
        
        status_text.text("✅ EAR batch analysis completed!")
        progress_bar.progress(1.0)
        
        # Show quick summary
        col1, col2, col3, col4 = st.columns(4)
        col1.success(f"✅ Detected: {detected_images}")
        col2.info(f"📊 Detection Rate: {detection_rate:.1f}%")
        col3.info(f"👁️ Avg EAR: {avg_ear:.4f}" if ear_values else "👁️ No EAR data")
        col4.warning(f"❌ Failed: {processed_images - detected_images}" if processed_images - detected_images > 0 else "✅ No Failures")
        
        if ear_values:
            st.success(f"Analysis completed! Processed {processed_images} images with {detection_rate:.1f}% detection rate. "
                      f"Average EAR: {avg_ear:.4f} (σ={ear_std:.4f})")
        else:
            st.warning("Analysis completed but no landmarks were detected in any images. Check your dataset quality.")
        
    except Exception as e:
        st.error(f"EAR batch test failed: {str(e)}")


# =============================================================================
# FULL PIPELINE TEST PAGE
# =============================================================================

def render_full_pipeline_test():
    """Render full pipeline integration test page."""
    st.header("Full Pipeline Integration Test")
    st.markdown("*Eye Analysis -> YOLO Classification -> Confidence Fusion -> NLP Correction*")
    
    # Initialize session state
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'pipeline_stats' not in st.session_state:
        st.session_state.pipeline_stats = {
            'frames': 0,
            'eye_times': [],
            'yolo_times': [],
            'fusion_times': [],
            'total_times': [],
            'confidences': []
        }
    if 'pipeline_nlp_history' not in st.session_state:
        st.session_state.pipeline_nlp_history = []
    
    config = SystemConfig()
    
    # Sidebar
    with st.sidebar:
        st.subheader("Pipeline Settings")
        alpha = st.slider("Alpha (YOLO weight)", 0.0, 1.0, config.alpha, 0.05)
        blink_threshold = st.slider("Blink Threshold", 0.1, 0.9, config.blink_threshold, 0.05)
        config.alpha = alpha
        config.blink_threshold = blink_threshold
    
    # Load all components
    st.subheader("Component Status")
    status_cols = st.columns(4)
    
    components_loaded = True
    
    with status_cols[0]:
        try:
            eye_analyzer = load_eye_analyzer()
            st.success("EyeAnalyzer")
        except Exception as e:
            st.error(f"EyeAnalyzer")
            components_loaded = False
    
    with status_cols[1]:
        try:
            yolo_classifier = load_yolo_classifier(config.yolo_model_path, config.use_gpu)
            st.success("YOLO")
        except Exception as e:
            st.error(f"YOLO")
            components_loaded = False
    
    with status_cols[2]:
        try:
            confidence_fusion = ConfidenceFusion(config.smoothing_window, config.ema_alpha)
            st.success("ConfidenceFusion")
        except Exception as e:
            st.error(f"Fusion")
            components_loaded = False
    
    with status_cols[3]:
        try:
            nlp_corrector = get_nlp_corrector()
            st.success("IndoBERT")
        except Exception as e:
            st.error(f"IndoBERT")
            components_loaded = False
    
    if not components_loaded:
        st.error("Some components failed to load. Cannot run full pipeline test.")
        return
    
    # Controls
    ctrl_cols = st.columns([1, 1, 1])
    with ctrl_cols[0]:
        if st.button("Start Pipeline", use_container_width=True, type="primary"):
            st.session_state.pipeline_running = True
            confidence_fusion.reset()
    with ctrl_cols[1]:
        if st.button("Stop", use_container_width=True):
            st.session_state.pipeline_running = False
    with ctrl_cols[2]:
        if st.button("Reset", use_container_width=True):
            st.session_state.pipeline_stats = {
                'frames': 0, 'eye_times': [], 'yolo_times': [],
                'fusion_times': [], 'total_times': [], 'confidences': []
            }
            st.session_state.pipeline_nlp_history = []
    
    # NLP Test section
    st.divider()
    nlp_cols = st.columns([3, 1])
    with nlp_cols[0]:
        nlp_input = st.text_input("Test NLP Correction:", placeholder="Enter text to correct...")
    with nlp_cols[1]:
        st.write("")
        st.write("")
        if st.button("Correct", use_container_width=True):
            if nlp_input:
                start = time.time()
                corrected = nlp_corrector.correct(nlp_input)
                nlp_time = (time.time() - start) * 1000
                st.session_state.pipeline_nlp_history.insert(0, {
                    'input': nlp_input, 'output': corrected, 'time': nlp_time
                })
    
    if st.session_state.pipeline_nlp_history:
        last = st.session_state.pipeline_nlp_history[0]
        st.markdown(f"**Result:** `{last['input']}` -> `{last['output']}` ({last['time']:.1f}ms)")
    
    st.divider()
    
    # Main display
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
    
    with col_metrics:
        st.subheader("Pipeline Output")
        state_display = st.empty()
        conf_display = st.empty()
        
        st.subheader("Timing Breakdown")
        timing_display = st.empty()
    
    # Stats section
    st.subheader("Performance Metrics")
    perf_placeholder = st.empty()
    
    # Video loop
    if st.session_state.pipeline_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("Cannot open webcam")
            st.session_state.pipeline_running = False
        else:
            try:
                while st.session_state.pipeline_running:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    total_start = time.time()
                    
                    # Stage 1: Eye Analysis
                    t1 = time.time()
                    eye_data, annotated = eye_analyzer.process_frame(frame, config)
                    eye_time = (time.time() - t1) * 1000
                    
                    # Stage 2: YOLO
                    t2 = time.time()
                    yolo_result = yolo_classifier.classify_dual_eye(
                        eye_data.left_crop, eye_data.right_crop
                    )
                    yolo_time = (time.time() - t2) * 1000
                    
                    # Stage 3: Fusion
                    t3 = time.time()
                    raw_conf = confidence_fusion.fuse(yolo_result, eye_data.normalized_ear, config.alpha)
                    smoothed_conf = confidence_fusion.smooth_ema(raw_conf)
                    fusion_time = (time.time() - t3) * 1000
                    
                    total_time = (time.time() - total_start) * 1000
                    
                    # Determine state
                    eye_state = EyeState.OPEN if smoothed_conf >= config.blink_threshold else EyeState.CLOSED
                    
                    # Update stats
                    stats = st.session_state.pipeline_stats
                    stats['frames'] += 1
                    stats['eye_times'].append(eye_time)
                    stats['yolo_times'].append(yolo_time)
                    stats['fusion_times'].append(fusion_time)
                    stats['total_times'].append(total_time)
                    stats['confidences'].append(smoothed_conf)
                    
                    # Keep last 100
                    for key in ['eye_times', 'yolo_times', 'fusion_times', 'total_times', 'confidences']:
                        if len(stats[key]) > 100:
                            stats[key] = stats[key][-100:]
                    
                    # Draw on frame
                    state_color = (0, 255, 0) if eye_state == EyeState.OPEN else (0, 0, 255)
                    cv2.putText(annotated, f"{eye_state.value.upper()}", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
                    cv2.putText(annotated, f"Conf: {smoothed_conf:.1%} | {total_time:.0f}ms", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Metrics
                    state_text = "Open" if eye_state == EyeState.OPEN else "Closed"
                    state_display.metric("Eye State", f"{state_text}")
                    conf_display.metric("Fused Confidence", f"{smoothed_conf:.1%}")
                    
                    timing_display.markdown(f"""
                    | Stage | Time |
                    |-------|------|
                    | Eye Analysis | {eye_time:.1f} ms |
                    | YOLO | {yolo_time:.1f} ms |
                    | Fusion | {fusion_time:.2f} ms |
                    | **Total** | **{total_time:.1f} ms** |
                    """)
                    
                    # Performance summary
                    perf_placeholder.markdown(f"""
                    | Metric | Eye Analysis | YOLO | Fusion | Total |
                    |--------|-------------|------|--------|-------|
                    | Mean | {np.mean(stats['eye_times']):.1f} ms | {np.mean(stats['yolo_times']):.1f} ms | {np.mean(stats['fusion_times']):.2f} ms | {np.mean(stats['total_times']):.1f} ms |
                    | Std | {np.std(stats['eye_times']):.1f} | {np.std(stats['yolo_times']):.1f} | {np.std(stats['fusion_times']):.2f} | {np.std(stats['total_times']):.1f} |
                    
                    **Frames:** {stats['frames']} | **Avg FPS:** {1000/np.mean(stats['total_times']):.1f} | **Avg Confidence:** {np.mean(stats['confidences']):.1%}
                    """)
                    
            finally:
                cap.release()
    else:
        video_placeholder.info("Click 'Start Pipeline' to begin full integration test")
        
        # Show final stats
        stats = st.session_state.pipeline_stats
        if stats['frames'] > 0:
            st.markdown(f"""
            ### Final Session Summary
            
            | Metric | Eye Analysis | YOLO | Fusion | Total |
            |--------|-------------|------|--------|-------|
            | Mean | {np.mean(stats['eye_times']):.1f} ms | {np.mean(stats['yolo_times']):.1f} ms | {np.mean(stats['fusion_times']):.2f} ms | {np.mean(stats['total_times']):.1f} ms |
            | Std | {np.std(stats['eye_times']):.1f} | {np.std(stats['yolo_times']):.1f} | {np.std(stats['fusion_times']):.2f} | {np.std(stats['total_times']):.1f} |
            
            **Total Frames:** {stats['frames']} | **Theoretical Max FPS:** {1000/np.mean(stats['total_times']):.1f}
            """)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Pipeline Tester - Eye-Blink Morse",
        page_icon="",
        layout="wide"
    )
    
    st.title("Eye-Blink Morse Code - Pipeline Tester")
    st.markdown("*Interactive testing for each system component*")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Test:",
        [
            "NLP Correction",
            "YOLO Classification",
            "EAR Analysis",
            "Full Pipeline"
        ]
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("**Author:** AI Lab - Tel-U")
    st.sidebar.markdown("**Date:** January 2026")
    
    # Render selected page
    if page == "NLP Correction":
        render_nlp_test()
    elif page == "YOLO Classification":
        render_yolo_test()
    elif page == "EAR Analysis":
        render_ear_test()
    elif page == "Full Pipeline":
        render_full_pipeline_test()


if __name__ == "__main__":
    main()
