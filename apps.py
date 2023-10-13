import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_most_similar_response(df, query, top_k=1):
    # Step 1: Prepare Data
    vectorizer = TfidfVectorizer()
    all_data = list(df['user_chat']) + [query]

    # Step 2: TF-IDF Vectorization
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Step 3: Compute Similarity
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)

    # Step 4: Sort and Pick Top k Responses
    sorted_indexes = similarity_scores.argsort()[0][-top_k:]
    
    # Fetch the corresponding responses from the DataFrame
    most_similar_responses = df.iloc[sorted_indexes]['response'].values
    return most_similar_responses

# Sample DataFrame with user_chat and response columns
data = {
    'user_chat': [
        'Tell me about the ingredients of Adobo',
        'What goes into making Sinigang?',
        'Explain the ingredients in Lumpia',
        'What makes Kare-Kare special in terms of ingredients?',
        'Can you describe the ingredients in Bicol Express?',
        'Give me the ingredients for Pancit Canton',
        'What is in a classic Halo-Halo dessert?',
        'How is Laing prepared and what are the key ingredients?',
        'Tell me about the ingredients in Lechon',
        'Explain the ingredients used in Chicken Adobo',
        'What are the main ingredients in Dinuguan',
        'Give me the ingredients for Pinakbet',
        'What is in a traditional Bibingka',
        'What ingredients are used in the preparation of Taho',
        'Can you describe the ingredients in Longganisa',
        'How do you make Turon and what are the key ingredients?',
        'What goes into making a traditional Kare-Kare',
        'Tell me about the ingredients of Tinolang Manok',
        'What makes Sinigang na Hipon special in terms of ingredients?',
        'Can you describe the ingredients in Ginataang Bilo-Bilo',
        'Give me the ingredients for Pancit Malabon',
        'What is in a classic Buko Salad',
        'How is La Paz Batchoy prepared and what are the key ingredients?',
        'Tell me about the ingredients in Tapsilog',
        'Explain the ingredients used in Munggo',
        'What are the main ingredients in Inihaw na Liempo',
        'Give me the ingredients for Ginisang Upo',
        'What is in a traditional Kakanin',
        'What ingredients are used in the preparation of Ukoy',
        'Can you describe the ingredients in Dinuguan at Puto',
        'How do you make Taho and what are the key ingredients?',
        'What goes into making Pancit Malabon',
        'Tell me about the ingredients of Kilawin',
        'What makes Puto Bumbong special in terms of ingredients?',
        'Can you describe the ingredients in Kinilaw na Tanigue',
        'Give me the ingredients for Kare-Kareng Gulay',
        'What is in a classic Espasol',
        'How is Adobong Pusit prepared and what are the key ingredients?',
        'Tell me about the ingredients in Ginisang Monggo',
        'What are the main ingredients in Lechon Manok',
        'Give me the ingredients for Ginataang Alimango',
        'What is in a traditional Kutsinta',
        'What ingredients are used in the preparation of Bibingkang Malagkit',
        'Can you describe the ingredients in Adobong Kangkong',
        'How do you make Ginisang Hipon and what are the key ingredients?',
        'What goes into making Bangusilog',
        'Tell me about the ingredients of Sinuglaw',
    ],
    'response': [
        'Adobo is a popular Filipino dish made with chicken or pork, soy sauce, vinegar, garlic, bay leaves, and black peppercorns.',
        'Sinigang is a sour soup prepared with pork, shrimp, or fish, and it typically includes tamarind, tomatoes, kangkong (water spinach), and radish.',
        'In Lumpia, you will find a mixture of ground pork, shrimp, vegetables (such as carrots and jicama), garlic, and soy sauce, all wrapped in a thin spring roll wrapper.',
        'Kare-Kare is known for its unique peanut sauce, and it\'s made with oxtail, tripes, and vegetables like eggplant, string beans, and banana blossoms.',
        'Bicol Express features pork or shrimp cooked in a spicy sauce made with coconut milk, shrimp paste, and chili peppers.',
        'Pancit Canton includes stir-fried noodles, shrimp, pork, vegetables like cabbage and bell peppers, soy sauce, and sometimes oyster sauce.',
        'Halo-Halo is a delightful dessert made with shaved ice, a mix of sweet ingredients like sweet potato, banana, jackfruit, and jellies, topped with leche flan and ube ice cream.',
        'Laing is made from dried taro leaves, coconut milk, shrimp paste, and chili peppers, creating a creamy and spicy dish.',
        'Lechon is a whole roasted pig seasoned with a mixture of herbs and spices, often including lemongrass, garlic, and salt.',
        'Chicken Adobo is similar to the classic Adobo but uses chicken as the main protein, cooked with soy sauce, vinegar, garlic, and spices.',
        'Dinuguan, also known as "chocolate meat," contains pork and offal simmered in a rich, savory sauce made with pork blood, vinegar, and spices.',
        'Pinakbet consists of mixed vegetables like squash, okra, and eggplant, along with shrimp paste and sometimes pork, creating a flavorful vegetable stew.',
        'Bibingka is a rice cake made from rice flour, coconut milk, and sugar, often topped with salted duck eggs and cheese.',
        'Taho is a sweet snack made from soft/silken tofu, arnibal (a caramel-like syrup made from brown sugar and sometimes with vanilla), and sago pearls.',
        'Longganisa is a Filipino sausage with variations in flavor across regions but typically includes ground pork, garlic, and vinegar.',
        'Turon is a sweet spring roll made with ripe bananas, brown sugar, and sometimes jackfruit, wrapped in a spring roll wrapper and fried until crispy.',
        'Traditional Kare-Kare is made with oxtail, tripes, and vegetables like eggplant, string beans, and banana blossoms, and it\'s served with a rich peanut sauce.',
        'Tinolang Manok is a chicken soup with ginger, garlic, green papaya, and chili leaves, simmered in a clear and flavorful broth.',
        'Sinigang na Hipon is a shrimp sour soup made with tamarind, tomatoes, radish, kangkong (water spinach), and green chili peppers.',
        'Ginataang Bilo-Bilo is a sweet coconut milk soup with rice balls (bilo-bilo), glutinous rice balls, and various root crops like sweet potato and taro.',
        'Pancit Malabon features thick rice noodles, shrimp, and a thick, savory sauce made from shrimp stock, fish sauce, and an array of seafood toppings.',
        'Buko Salad is a refreshing dessert with young coconut strips, fruit cocktail, cream, and condensed milk, often topped with grated cheese.',
        'La Paz Batchoy is a noodle soup made with pork, beef, and chicken, along with miki noodles, garlic, and crushed chicharrón (pork cracklings).',
        'Tapsilog is a popular Filipino breakfast dish consisting of beef tapa (marinated and cured beef), garlic fried rice, and fried eggs.',
        'Munggo is a savory mung bean stew made with mung beans, shrimp, and malunggay leaves, often flavored with fish sauce and garlic.',
        'Inihaw na Liempo is grilled pork belly marinated with a mixture of garlic, soy sauce, and calamansi (Philippine lime).',
        'Ginisang Upo is a simple sautéed dish made with bottle gourd (upo), shrimp, tomatoes, and fish sauce.',
        'Kakanin is a variety of Filipino rice cakes, typically made from glutinous rice, coconut milk, and sugar, and can include kutsinta, sapin-sapin, and bibingka.',
        'Ukoy is a crispy shrimp and vegetable fritter made from small shrimp, bean sprouts, and sweet potato, bound together with a batter.',
        'Dinuguan at Puto is a unique combination of Dinuguan (pork blood stew) and puto (steamed rice cake), often served together.',
        'Taho is made with silken tofu, arnibal (a sweet syrup made from brown sugar), and sago pearls, creating a sweet and silky dessert.',
        'Pancit Malabon is made with thick rice noodles, shrimp, and a thick sauce made from shrimp stock, fish sauce, and a variety of seafood toppings.',
        'Kilawin is a Filipino ceviche made with raw fish or seafood marinated in vinegar, calamansi juice, onions, ginger, and chili peppers.',
        'Puto Bumbong is a purple rice cake steamed in bamboo tubes, made from a special variety of glutinous rice, muscovado sugar, and grated coconut.',
        'Kinilaw na Tanigue is a dish made with fresh mackerel fish marinated in vinegar, calamansi juice, ginger, and onions, typically served with coconut milk.',
        'Kare-Kareng Gulay is a vegetarian version of Kare-Kare, made with various vegetables like eggplant, string beans, and banana blossoms, served with a peanut sauce.',
        'Espasol is a cylindrical rice cake made from glutinous rice flour and coconut milk, rolled in toasted rice flour and sugar.',
        'Adobong Pusit is a squid dish cooked in soy sauce, vinegar, garlic, onions, and chili peppers, creating a savory and slightly tangy flavor.',
        'Ginisang Monggo is a mung bean soup made with mung beans, pork, and shrimp, flavored with fish sauce and garlic.',
        'Lechon Manok is a roasted chicken marinated with a blend of spices and aromatics, resulting in a flavorful and crispy skin.',
        'Ginataang Alimango is a crab dish cooked in coconut milk with ginger, chili peppers, and sometimes green papaya.',
        'Kutsinta is a steamed rice cake made from rice flour, sugar, and lye water, giving it a distinctive chewy texture and a caramelized topping.',
        'Bibingkang Malagkit is a sticky rice cake made with glutinous rice, coconut milk, sugar, and often topped with coconut custard and sometimes with jackfruit.',
        'Adobong Kangkong is a vegetable dish made with water spinach (kangkong) sautéed in garlic, soy sauce, vinegar, and often with shrimp paste.',
        'Ginisang Hipon is a shrimp sauté made with shrimp, garlic, tomatoes, and fish sauce, creating a simple and tasty dish.',
        'Bangusilog is a breakfast dish featuring bangus (milkfish) marinated and fried, served with garlic fried rice and fried eggs.',
        'Sinuglaw is a combination of sinugba (grilled pork) and kinilaw (ceviche), often featuring pork, fish, vinegar, calamansi, and chili peppers.',
    ]
}

df = pd.DataFrame(data)

st.title("Filipino Dishes Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    responses = get_most_similar_response(df, prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        for response in responses:
            st.markdown(f"{response}")

    # Add assistant response to chat history
    for response in responses:
        st.session_state.messages.append({"role": "assistant", "content": f"Echo: {response}"})