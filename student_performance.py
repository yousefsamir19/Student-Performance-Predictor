import pickle

# Save the trained linear regression model
with open("linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the polynomial regression model
with open("poly_model.pkl", "wb") as f:
    pickle.dump(model_poly, f)

# Also save the polynomial transformer (for input processing)
with open("poly_transformer.pkl", "wb") as f:
    pickle.dump(poly, f)