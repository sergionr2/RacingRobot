from torch.utils.cpp_extension import load
custom_cnn = load(name="custom_cnn", sources=["custom_cnn.cpp"], verbose=True)
help(custom_cnn)
