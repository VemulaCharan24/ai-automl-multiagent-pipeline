from inference.inference_engine import analyze_prompt

if __name__ == "__main__":
    while True:
        text = input("Enter your ML task: ")
        result = analyze_prompt(text)
        print(result)
