import pandas as pd
import mysql.connector
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from flask import Flask, render_template, request

conn=mysql.connector.connect(
    
)