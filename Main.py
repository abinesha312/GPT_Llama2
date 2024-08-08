import gradio as gr
from pages import login_page, register_page, chatbot_page
from auth import authenticate, create_user
from chatbot import respond

def change_page(page):
    return {
        "login": page == "login",
        "register": page == "register",
        "chatbot": page == "chatbot"
    }

with gr.Blocks() as demo:
    page_state = gr.State("login")
    
    with gr.Group() as login_group:
        username, password, login_button = login_page()
        new_user_button = gr.Button("New User? Register here")

    with gr.Group(visible=False) as register_group:
        reg_username, reg_password, reg_verify_password, reg_mobile, reg_firstname, reg_lastname, register_button = register_page()
        back_to_login_button = gr.Button("Back to Login")

    with gr.Group(visible=False) as chatbot_group:
        chatbot, msg, clear = chatbot_page()
        logout_button = gr.Button("Logout")

    def login(username, password):
        if authenticate(username, password):
            gr.Info("Login successful!")
            return "chatbot"
        else:
            gr.Warning("Invalid username or password!")
            return "login"

    def register(u, p, vp, m, fn, ln):
        result = create_user(u, p, vp, m, fn, ln)
        if result == "User created successfully":
            gr.Info("Registration successful! Please log in.")
            return "login"
        else:
            gr.Warning(f"Registration failed: {result}")
            return "register"

    def update_page(new_page):
        visibility = change_page(new_page)
        return (
            gr.update(visible=visibility["login"]),
            gr.update(visible=visibility["register"]),
            gr.update(visible=visibility["chatbot"]),
            new_page
        )

    new_user_button.click(
        lambda: "register",
        outputs=[page_state]
    ).then(
        update_page,
        inputs=[page_state],
        outputs=[login_group, register_group, chatbot_group, page_state]
    )

    back_to_login_button.click(
        lambda: "login",
        outputs=[page_state]
    ).then(
        update_page,
        inputs=[page_state],
        outputs=[login_group, register_group, chatbot_group, page_state]
    )

    logout_button.click(
        lambda: "login",
        outputs=[page_state]
    ).then(
        update_page,
        inputs=[page_state],
        outputs=[login_group, register_group, chatbot_group, page_state]
    )

    login_button.click(
        login,
        inputs=[username, password],
        outputs=[page_state]
    ).then(
        update_page,
        inputs=[page_state],
        outputs=[login_group, register_group, chatbot_group, page_state]
    )

    register_button.click(
        register,
        inputs=[reg_username, reg_password, reg_verify_password, reg_mobile, reg_firstname, reg_lastname],
        outputs=[page_state]
    ).then(
        update_page,
        inputs=[page_state],
        outputs=[login_group, register_group, chatbot_group, page_state]
    )

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=True to create a public link