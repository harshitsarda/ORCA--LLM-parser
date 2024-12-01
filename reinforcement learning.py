import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample paragraph and questions
paragraph = "The company launched its new software with features like data encryption and automatic backups. Pricing information is available, but there's no mention of customer support options."
questions = [
    "Does it mention pricing?",
    "Is customer support discussed?",
    "Does it talk about data encryption?",
]
ground_truth = ["Yes", "No", "Yes"]

# Reward function (simplified for demonstration)
def reward_function(answer, explanation, ground_truth, question):
    reward = 0
    # Reward for correct answer
    if answer == ground_truth:
        reward += 10
    # Explanation quality (example logic, can be improved)
    if explanation and ground_truth.lower() in explanation.lower():
        reward += 5
    # Penalty for incorrect answers or vague explanations
    if answer != ground_truth or not explanation:
        reward -= 5
    return reward

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 5

for epoch in range(num_epochs):
    total_reward = 0
    for i, question in enumerate(questions):
        # Encode input
        input_text = f"Paragraph: {paragraph}\nQuestion: {question}\nAnswer and Explanation:"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Generate response
        outputs = model.generate(inputs["input_ids"], max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(respnse)
        # Parse response
        try:
            answer, explanation = response.split("Explanation: ")
        except ValueError:
            answer, explanation = response, ""
        
        # Calculate reward
        reward = reward_function(answer.strip(), explanation.strip(), ground_truth[i], question)
        total_reward += reward
        
        # Update model using the reward
        loss = -torch.tensor(reward, dtype=torch.float, requires_grad=True)  # Ensure dtype is float
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch + 1}, Total Reward: {total_reward}")

print("Training complete!")
