import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable not set.")
    st.stop()

def generate_restaurant_name_and_items(cuisine):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    
    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest only 10 menu items for {restaurant_name}. Return it as a comma separated string."
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
    
    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )
    
    response = chain({'cuisine': cuisine})
    
    return response

st.title("Restaurant Name Generator 🍔")
cuisine = st.sidebar.selectbox("Pick a cuisine", ("Indian", "Italian", "Mexican", "Arabic", "American", "Chinese", "French"))

if cuisine:
    response = generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].strip().split(',')
    st.write("## MENU ITEMS ##")
    for item in menu_items:
        st.write(f"- {item.strip()}")
