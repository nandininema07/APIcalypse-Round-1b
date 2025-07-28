# Approach and Algorithm Explanation

Our solution is a smart system that understands documents. It is designed to find and rank the most important parts of complex PDF files. Instead of just searching for keywords, our system works to truly understand what a user wants to achieve, based on their described **persona** and **job to be done**. To do this, it uses a combination of smart technologies that analyze both the words on the page and the page's visual design.

Our process works in four main steps:

1.  **Understanding the Context**: First, we break down the user's request. We use a lightweight model called **BERT-Tiny** to turn the text into a numerical format that the computer can understand. From this, we pick out the core **"domain terms"**—the words most related in meaning to the user's goal. These core terms then help us create a broader list of **"context keywords"** that will guide the ranking process later. This gives us a clear picture of what the user needs.

2.  **Finding Sections Using Layout**: To read the documents correctly, we use **LayoutLMv3**, a powerful model that looks at each PDF page as if it were an image. This allows it to understand the document's structure by seeing visual clues like font sizes and text placement. This approach works well even on pages with complicated designs. We then check the extracted sections to filter out any irrelevant bits.

3.  **Ranking Sections with a Boost**: Each section is converted into a numerical representation. We then calculate an initial relevance score by comparing how similar the section is to the user's request. The key part of our approach is our **"boosting" mechanism**: this initial score gets increased if the section contains the context keywords we found earlier. A section with more of these keywords gets a much higher score, which better reflects its importance.

4.  **Summarizing and Delivering Results**: Finally, the sections are sorted from most to least important based on their final boosted scores. For the top-ranked sections, we look closer to pull out the most critical sentences, creating a short and useful summary. The final result is a neatly organized JSON file that contains the ranked sections and their key information.