import google.generativeai as genai
genai.configure(api_key='AIzaSyAUlcUPDs-qgDdRWHgmdi8hgHvj3btAQ2s')
model = genai.GenerativeModel('gemini-pro')
print(model.generate_content('안녕').text)