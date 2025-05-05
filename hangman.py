import turtle as trtl
from functools import partial
import random
import time

words = [
    "which",
    "there",
    "their",
    "about",
    "would",
    "these",
    "other",
    "words",
    "could"
]

print("guess one word from this list below")
for word in words:
    print(word)

##my_file = open("words.txt", "r")
my_file=open("C:\pycharm\SnehaProject\words.txt","r")
data = my_file.read()
words = data.split("\n")
my_file.close()
select_word = words[random.randint(0, len(words) - 1)].lower()
painter = trtl.Turtle()
painter.fillcolor("white")
painter.color("white")
text = trtl.Turtle()
wn = trtl.Screen()
wn.bgcolor("black")
text.fillcolor("white")
text.color("white")
painter.pensize(2)
FONT = ("Times New Roman", 70, "normal")
select_word_letters = []
for i in range(0, len(select_word)):
    select_word_letters.append(select_word[i])
guessed_word_letters = ["_", "_", "_", "_", "_"]
times_guessed_wrong = 0


def draw_head():
    painter.penup()
    painter.goto(0, 200)
    painter.pendown()
    painter.circle(50)
    painter.penup()
    painter.left(90)
    painter.forward(60)
    painter.left(90)
    painter.forward(20)
    painter.pendown()
    painter.begin_fill()
    painter.circle(4)
    painter.end_fill()
    painter.penup()
    painter.left(170)
    painter.forward(35)
    painter.pendown()
    painter.begin_fill()
    painter.circle(4)
    painter.end_fill()
    painter.penup()
    painter.fillcolor("black")


def draw_body():
    painter.goto(0, 200)
    painter.pendown()
    painter.goto(0, 100)
    painter.penup()


def draw_left_arm():
    painter.goto(0, 200)
    painter.pendown()
    painter.goto(-35, 120)
    painter.penup()


def draw_right_arm():
    painter.goto(0, 200)
    painter.pendown()
    painter.goto(35, 120)
    painter.penup()


def draw_left_leg():
    painter.goto(0, 100)
    painter.pendown()
    painter.goto(-35, 20)
    painter.penup()


def draw_right_leg():
    painter.goto(0, 100)
    painter.pendown()
    painter.goto(35, 20)
    painter.penup()


def _onkeypress(self, fun, key=None):
    if fun is None:
        if key is None:
            self.cv.unbind("<KeyPress>", None)
        else:
            self.cv.unbind("<KeyPress-%s>" % key, None)
    elif key is None:
        def eventfun(event):
            fun(event.char)

        self.cv.bind("<KeyPress>", eventfun)
    else:
        def eventfun(event):
            fun()

        self.cv.bind("<KeyPress-%s>" % key, eventfun)


def write_letters():
    # text.hideturtle()
    guessed_word = "".join(guessed_word_letters)
    text.clear()
    text.speed(0)
    text.penup()
    text.goto(-80, -300)
    text.write(guessed_word, font=FONT)
    text.goto(-80, -150)
    text.goto(-90, 350)
    # text.write("Guess the word! " , font=("Times New Roman", 35, "normal"))


def letter(character):
    global times_guessed_wrong
    if character.lower() in select_word_letters:
        for i in range(0, len(select_word_letters)):
            if character.lower() == select_word_letters[i]:
                guessed_word_letters[i] = character.lower()
    else:
        times_guessed_wrong += 1
        if times_guessed_wrong == 1:
            draw_head()
        elif times_guessed_wrong == 2:
            draw_body()
        elif times_guessed_wrong == 3:
            draw_left_arm()
        elif times_guessed_wrong == 4:
            draw_right_arm()
        elif times_guessed_wrong == 5:
            draw_left_leg()
        elif times_guessed_wrong == 6:
            draw_right_leg()
            time.sleep(1)
            game_over()
            time.sleep(5)
            wn.bye()
    write_letters()

    if guessed_word_letters == select_word_letters:
        win_game()
        time.sleep(3)
        wn.bye()


def game_over():
    text.clear()
    painter.clear()
    text.color("red")
    text.goto(-430, 0)
    game_over_text = "Game Over! You have lost."
    game_over_text2 = "The word was {}".format(select_word)
    text.write(game_over_text, font=("Times New Roman", 80, "normal"))
    text.goto(-230, -100)
    text.write(game_over_text2, font=("Times New Roman", 50, "normal"))


def win_game():
    text.clear()
    painter.clear()
    text.goto(-430, 0)
    game_over_text = "Great Job!! You won!"
    game_over_text2 = "The word was {}!!".format(select_word)
    text.write(game_over_text, font=("Times New Roman", 80, "normal"))
    text.goto(-230, -100)
    text.write(game_over_text2, font=("Times New Roman", 50, "normal"))


write_letters()
# draw_head()
wn._onkeypress = partial(_onkeypress, wn)
wn.onkeypress(letter)
wn.listen()
wn.mainloop()