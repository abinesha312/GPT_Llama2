# import torch
# import mesop as me
# import mesop.labs as mel
# from AuthFLD.Auth import Auth
# import pandas as pd
# import env as en
# import Model.ChatInterface as chatbt

import torch
import mesop as me
import mesop.labs as mel
from AuthFLD.Auth import Auth
import env as en
import Model.ChatInterface as chatbt

global_llm = None
global_db = None


def free_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

@me.stateclass
class AppState:
    session_id: str = ""
    user_id: str = ""
    username: str = ""
    error_message: str = ""

@me.stateclass
class RegisterState:
    username: str = ""
    password: str = ""
    confirm_password: str = ""
    name: str = ""
    email: str = ""
    contact_number: str = ""
    error_message: str = ""

@me.stateclass
class State:
    selected_user: str = "No user selected."


def is_authenticated():
    state = me.state(AppState)
    return bool(state.session_id)

@me.page(path="/", title="UNT Chatbot")
def main_page():
    if not is_authenticated():
        me.navigate("/login")
    else:
        chatbot_page()


@me.stateclass
class LoginState:
    username: str = ""
    password: str = ""
    error_message: str = ""
    
@me.stateclass
class LoginState:
    username: str = ""
    password: str = ""
    error_message: str = ""

@me.page(path="/users", title="List of Users")
def Load_Users():
    user_keys = en.redis_client.keys("user:*")
    data = []
    for key in user_keys:
        user_data = en.redis_client.hgetall(key)
        user_dict = {k.decode(): v.decode() for k, v in user_data.items() if k.decode() != 'password'}
        data.append(user_dict)
     
    df = pd.DataFrame(data)
    columns = ['username', 'name', 'email', 'contact_number', 'user_id',]
    df = df.reindex(columns=columns)
    
    with me.box(style=me.Style(padding=me.Padding.all(10), width=800)):
        me.table(
            df,
            header=me.TableHeader(sticky=True),
            columns={
                "username": me.TableColumn(sticky=True),
                "user_id": me.TableColumn(sticky=True),
            },
        )

@me.page(
    path="/login",
    title="Login",
)
def login_page():
    state = me.state(LoginState)
    app_state = me.state(AppState)

    with me.box(style=me.Style(padding=me.Padding.all(30))):
        me.text("Login", type="headline-4")

        with me.box(style=me.Style(margin=me.Margin(top=20))):
            me.input(
                label="Username",
                on_blur=lambda e: setattr(state, 'username', e.value),
                style=me.Style(width="100%")
            )

        with me.box(style=me.Style(margin=me.Margin(top=15))):
            me.input(
                label="Password",
                type="password",
                on_blur=lambda e: setattr(state, 'password', e.value),
                style=me.Style(width="100%")
            )

        with me.box(style=me.Style(margin=me.Margin(top=20), display="flex", justify_content="space-between")):
            me.button(
                "Login",
                on_click=handle_login,
                color="primary",
                type="raised"
            )
            me.button(
                "Register",
                on_click=lambda _: me.navigate("/register"),
                type="flat"
            )

        if state.error_message:
            with me.box(style=me.Style(margin=me.Margin(top=10))):
                me.text(
                    state.error_message,
                    style=me.Style(color="red")
                )


def handle_login(e: me.ClickEvent):
    state = me.state(LoginState)
    app_state = me.state(AppState)
    
    success, message, session_id, user_id = Auth.login(en.redis_client, state.username, state.password)
    if success:
        app_state.session_id = session_id
        app_state.user_id = user_id
        app_state.username = state.username
        me.navigate("/chatbot")
    else:
        state.error_message = message


@me.page(path="/register", title="Register")
def register_page():
    state = me.state(RegisterState)

    with me.box(style=me.Style(
        padding=me.Padding.all(30),
        max_width="900px",
        margin=me.Margin(left="auto", right="auto")
    )):
        me.text("Register", type="headline-4")

        with me.box(style=me.Style(
            margin=me.Margin.all(20),
            display="flex",
            justify_content="space-between",
            flex_wrap="wrap"
        )):
            me.input(
                label="Username",
                on_blur=lambda e: setattr(state, 'username', e.value),
                style=me.Style(width="48%", min_width="400px")
            )
            me.input(
                label="Name",
                on_blur=lambda e: setattr(state, 'name', e.value),
                style=me.Style(width="48%", min_width="400px")
            )

        with me.box(style=me.Style(
            margin=me.Margin.all(15),
            display="flex",
            justify_content="space-between",
            flex_wrap="wrap"
        )):
            me.input(
                label="Password",
                type="password",
                on_blur=lambda e: setattr(state, 'password', e.value),
                style=me.Style(width="48%", min_width="400px")
            )
            me.input(
                label="Confirm Password",
                type="password",
                on_blur=lambda e: setattr(state, 'confirm_password', e.value),
                style=me.Style(width="48%", min_width="400px")
            )

        with me.box(style=me.Style(
            margin=me.Margin.all(15),
            display="flex",
            justify_content="space-between",
            flex_wrap="wrap"
        )):
            me.input(
                label="Email",
                on_blur=lambda e: setattr(state, 'email', e.value),
                style=me.Style(width="48%", min_width="400px")
            )
            me.input(
                label="Contact Number",
                on_blur=lambda e: setattr(state, 'contact_number', e.value),
                style=me.Style(width="48%", min_width="400px")
            )

        with me.box(style=me.Style(
            margin=me.Margin.all(20),
            display="flex",
            justify_content="space-between"
        )):
            me.button(
                "Register",
                on_click=lambda _: handle_register(state),
                color="primary",
                type="raised"
            )
            me.button(
                "Back to Login",
                on_click=lambda _: me.navigate("/login"),
                type="flat"
            )

        if state.error_message:
            with me.box(style=me.Style(margin=me.Margin.all(10))):
                me.text(
                    state.error_message,
                    style=me.Style(color="red")
                )

def handle_register(state: RegisterState):
    if state.password != state.confirm_password:
        state.error_message = "Passwords do not match"
    else:
        success, message = Auth.register(en.redis_client, state.username, state.password, state.name, state.email, state.contact_number)
        if success:
            state.error_message = ""
            me.navigate("/login")
        else:
            state.error_message = message


@me.page(path="/logout", title="Logout")
def logout_page():
    state = me.state(AppState)
    if state.session_id:
        Auth.logout(en.redis_client, state.session_id)
    state.session_id = ""
    state.user_id = ""
    state.username = ""
    me.navigate("/login")

@me.page(path="/chatbot", title="UNT Chatbot")
def chatbot_page():
    # if not is_authenticated():
    #     me.navigate("/login")
    # else:
    # state = me.state(AppState)
    # me.text(f"Welcome, {state.username} !", type="headline-5")
    mel.chat(transform, title="Welcome to University of North Texas GPTSystem", bot_user="Chatbot")
    # me.button("Logout", on_click=lambda _: me.navigate("/logout"), type="flat")

def transform(user_input: str, history: list) -> str:
    if not history:
        history = []
    chat_history = [chatbt.ChatMessage(**msg) if isinstance(msg, dict) else msg for msg in history]
    try:
        _, updated_history = chatbt.handle_user_message(user_input, chat_history)
        return updated_history[-1].content if updated_history else ""
    except Exception as e:
        print(f"Error in transform: {e}")
        return f"An error occurred: {str(e)}"

# def transform(user_input: str, history: list) -> str:
#     if not history:
#         history = []
#     chat_history = [chatbt.ChatMessage(**msg) if isinstance(msg, dict) else msg for msg in history]
#     _, updated_history = chatbt.handle_user_message(user_input, chat_history)
#     return updated_history[-1].content if updated_history else ""

def main():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    chatbt.load_llm()
    chatbt.load_db()    
    print("Model and database loaded successfully")
    me.run()

if __name__ == "__main__":
    main()