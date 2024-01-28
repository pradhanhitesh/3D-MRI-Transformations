<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    /* Apply some basic styling to the container */
    .container {
      display: flex;
    }

    /* Set equal width for both columns */
    .column {
      flex: 1;
    }

    /* Optional: Add some padding or styling to the columns */
    .column img {
      width: 100%;
      height: auto;
      display: block;
    }

    .column p {
      text-align: justify;
    }
  </style>
</head>
<body>

  <div class="container">
    <!-- First column with image -->
    <div class="column">
      <img src="./data/Brain_Brodmann_blend.gif" alt="Your Image">
    </div>

    <!-- Second column with text -->
    <div class="column">
      <p>Your text goes here. Lorem ipsum dolor sit amet, consectetur adipiscing elit...</p>
    </div>
  </div>

</body>
</html>