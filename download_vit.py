from transformers import LevitModel, ViTModel
 
 
model = LevitModel.from_pretrained("facebook/levit-192")
model.save_pretrained("local_models/levit")

