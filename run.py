from irisFlaskApp.app import app
from flask import Flask, request, render_template,redirect, url_for

import numpy as np
import pandas as pd
from sklearn import tree
if __name__ == "__main__":
    app.run()