import numpy as np
import torch

# Define the path to the clip.npy file (make sure it's in the right place)
clip_file_path = 'vision_features/clip.npy'  # Suppose this script is at the same level as the vision_features directory, or provides the full path

print(f"Attempting to load: {clip_file_path}")

try:
    # Scenario 1: Use numpy.load (typically used for .npy files)
    print("\n--- Loading with numpy.load ---")
    # allow_pickle=True is used to handle cases where a .npy file may contain a Python object, such as a dictionary
    clip_features_data_np = np.load(clip_file_path, allow_pickle=True)

    print(f"Type of data loaded by numpy: {type(clip_features_data_np)}")

    if isinstance(clip_features_data_np, np.ndarray):
        print(f"Shape of numpy array: {clip_features_data_np.shape}")
        if clip_features_data_np.ndim > 0 and clip_features_data_np.shape[0] > 0:
            print(f"Data type of numpy array elements: {clip_features_data_np.dtype}")
            print(
                f"Shape of the first element (e.g., one image's features): {clip_features_data_np[0].shape if hasattr(clip_features_data_np[0], 'shape') else 'N/A, likely a scalar or complex object'}")
            # Assume that the last dimension is the feature dimension
            if clip_features_data_np.ndim >= 2:
                print(f"Assumed feature dimension (last axis): {clip_features_data_np.shape[-1]}")

    elif isinstance(clip_features_data_np, dict):
        print("Data loaded by numpy is a dictionary.")
        print(f"Number of items in dictionary: {len(clip_features_data_np)}")
        if len(clip_features_data_np) > 0:
            first_key = list(clip_features_data_np.keys())[0]
            first_value = clip_features_data_np[first_key]
            print(f"Sample key: {first_key}")
            print(f"Type of value for sample key: {type(first_value)}")
            if hasattr(first_value, 'shape'):
                print(f"Shape of value for sample key: {first_value.shape}")
                if first_value.ndim > 0:
                    print(f"Assumed feature dimension (last axis of value): {first_value.shape[-1]}")
            if hasattr(first_value, 'dtype'):
                print(f"Dtype of value for sample key: {first_value.dtype}")


    elif hasattr(clip_features_data_np, 'item') and isinstance(clip_features_data_np.item(), dict):
        print("Data loaded by numpy is a 0-dim array containing a dictionary. Accessing dict with .item()")
        actual_dict = clip_features_data_np.item()
        print(f"Number of items in dictionary: {len(actual_dict)}")
        if len(actual_dict) > 0:
            first_key = list(actual_dict.keys())[0]
            first_value = actual_dict[first_key]
            print(f"Sample key: {first_key}")
            print(f"Type of value for sample key: {type(first_value)}")
            if hasattr(first_value, 'shape'):
                print(f"Shape of value for sample key: {first_value.shape}")
                if first_value.ndim > 0:
                    print(f"Assumed feature dimension (last axis of value): {first_value.shape[-1]}")
            if hasattr(first_value, 'dtype'):
                print(f"Dtype of value for sample key: {first_value.dtype}")


except Exception as e:
    print(f"Error loading with numpy.load: {e}")

# (Optional Scenario 2: Try using torch.load (if the .npy is saved with torch.save, or you want to see how torch handles it)
# print("\n--- Loading with torch.load ---")
# try:
#     # map_location='cpu' ensures that the file loads on the CPU even if it is saved on the GPU
#     clip_features_data_torch = torch.load(clip_file_path, map_location='cpu')
#     print(f"Type of data loaded by torch: {type(clip_features_data_torch)}")

#     if isinstance(clip_features_data_torch, torch.Tensor):
#         print(f"Shape of torch tensor: {clip_features_data_torch.shape}")
#         print(f"Data type of torch tensor: {clip_features_data_torch.dtype}")
#     elif isinstance(clip_features_data_torch, dict):
#         print("Data loaded by torch is a dictionary.")
#         print(f"Number of items in dictionary: {len(clip_features_data_torch)}")
#         if len(clip_features_data_torch) > 0:
#             first_key = list(clip_features_data_torch.keys())[0]
#             first_value = clip_features_data_torch[first_key]
#             print(f"Sample key: {first_key}")
#             print(f"Type of value for sample key: {type(first_value)}")
#             if hasattr(first_value, 'shape'):
#                 print(f"Shape of value for sample key: {first_value.shape}")
#             if hasattr(first_value, 'dtype'):
#                 print(f"Dtype of value for sample key: {first_value.dtype}")

# except Exception as e:
#     print(f"Error loading with torch.load: {e}")