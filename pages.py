import gradio as gr

def login_page():
    with gr.Group() as login_group:
        gr.Markdown("# Login")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_button = gr.Button("Login")
    return username, password, login_button

def register_page():
    with gr.Group() as register_group:
        gr.Markdown("# Register New User")
        reg_username = gr.Textbox(label="Username")
        reg_password = gr.Textbox(label="Password", type="password")
        reg_verify_password = gr.Textbox(label="Verify Password", type="password")
        reg_mobile = gr.Textbox(label="Mobile Number")
        reg_firstname = gr.Textbox(label="First Name")
        reg_lastname = gr.Textbox(label="Last Name")
        register_button = gr.Button("Register")
    return reg_username, reg_password, reg_verify_password, reg_mobile, reg_firstname, reg_lastname, register_button

def chatbot_page():
    with gr.Group() as chatbot_group:
        gr.Markdown("# AI Chatbot")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Type your message here...")
        clear = gr.Button("Clear Chat")
    return chatbot, msg, clear