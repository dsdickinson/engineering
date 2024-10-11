# Coding Exercise: Decoding a Secret Message

In this exercise, you will write code to solve a problem. Your code must be in either Python or JavaScript—solutions in other languages will not be accepted! You can write your code using any IDE you want.

## Problem

You are given the Google Doc **_data.doc_** which contains a list of Unicode characters and their positions in a 2D grid. Your task is to write a function that takes in the URL for such a Google Doc as an argument, retrieves and parses the data in the document, and prints the grid of characters. When printed in a fixed-width font, the characters in the grid will form a graphic showing a sequence of uppercase letters, which is the secret message.

- The document specifies the Unicode characters in the grid, along with the x- and y-coordinates of each character.
- The minimum possible value of these coordinates is 0. There is no maximum possible value, so the grid can be arbitrarily large.
- Any positions in the grid that do not have a specified character should be filled with a space character.
- You can assume the document will always have the same format as the example document linked above.

For example, the simplified example document **_example_F.doc_** draws out the letter 'F':

█▀▀▀

█▀▀ 

█   

Note that the coordinates (0, 0) will always correspond to the same corner of the grid as in this example, so make sure to understand in which directions the x- and y-coordinates increase.

## Specifications
- Your code must be written in Python (preferred) or JavaScript.
- You may use external libraries.
- You may write helper functions, but there should be one function that:

1. Takes in one argument, which is a string containing the URL for the Google Doc with the input data, AND
2. When called, prints the grid of characters specified by the input data, displaying a graphic of correctly oriented uppercase letters.
