from django.shortcuts import render, HttpResponse
import datetime
from transformers import TFAutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration, T5ForConditionalGeneration
from jinja2 import Template



# Create your views here.
def welcome(request):
    if request.method == 'POST':
        # Handle the form submission and execute the desired function
        prompt = request.POST.get('prompt')
        
        # Call your custom function with the submitted prompt
        generated_text = generate_text(prompt)
        
        # Return the result in an HttpResponse or render a template
        return HttpResponse(f'Generated Text: {generated_text}')

    return render(request,'index.html')


def generate_text(prompt):
    
    my_date = datetime.date(2023, 9, 18)
    formatted_date = my_date.strftime("%B %d, %Y")
    model_name = "gpt2"  

# Initialize the model and tokenizer
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text generation pipeline using your model and tokenizer
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Input text for generation
    prompt = prompt

# Generate text
    generated_text = text_generator(prompt, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92)

# Print the generated text
    text = generated_text[0]['generated_text']
    tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    model = BartForConditionalGeneration.from_pretrained("ainize/bart-base-cnn")

# Encode Input Text
    input_text = text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate Summary Text Ids
    summary_text_ids = model.generate(
    input_ids=input_ids,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    length_penalty=2.0,
    max_length=142,
    min_length=56,
    num_beams=4,
    )
# Decoding Text
    summary = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    tokenizer = AutoTokenizer.from_pretrained("czearing/article-title-generator")
    model = T5ForConditionalGeneration.from_pretrained("czearing/article-title-generator")

    input_text = text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    template_file = './templates/blog_template.jinja2'

    with open(template_file, 'r',encoding='utf-8') as file:
        template_content = file.read()

# Define context variables
    context = {
        "title": title,
        "author": "Raviteja, Engineer",
        "date": formatted_date ,
        "prompt":prompt,
        "summary": summary,
        "blogText":text,
        "intro": "intro",
        
        # Add other variables as needed
    }
    template = Template(template_content)

    # Render the template with context
    rendered_content = template.render(**context)

    # Specify the output HTML file path

    output_html_file = 'a33483.html'

    # Write the rendered content to the output HTML file
    with open(output_html_file, 'w', encoding = 'utf-8') as output_file:
        output_file.write(rendered_content)

    print(f'HTML file "{output_html_file}" has been created with the rendered content.')    

