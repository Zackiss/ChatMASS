import deepl

auth_key = "f63c02c5-f056-..."  # Replace with your key
translator = deepl.Translator(auth_key)


    result = translator.translate_text(text, target_lang="ZH")
    print(result.text)



