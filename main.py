from functions import(load_environment_variables, get_pdf_file, extract_pdf_content,
    separate_content_types, get_images_base64, create_summary_chain,
    create_image_description_chain, describe_images, safe_batch_summarize,
    setup_retriever, create_qa_chain, display_base64_image)
import os


class PDFProcessor:
    def __init__(self):
        self.env_vars = load_environment_variables()
        self.retriever = None
        self.qa_chain = None
        self.file_content = get_pdf_file()

        if self.file_content:
            self.process_pdf_content()
            self.ask_questions_loop()

    def process_pdf_content(self):
        """Process the PDF content"""
        if not self.file_content:
            print("No PDF content available to process")
            return

        try:
            chunks = extract_pdf_content(file_content=self.file_content)
            tables, texts = separate_content_types(chunks)
            images = get_images_base64(chunks)

            summary_chain = create_summary_chain(self.env_vars["GROQ_API_KEY"])
            image_chain = create_image_description_chain(self.env_vars["GEMINI_API_KEY"])

            print("Generating text summaries...")
            text_summaries = safe_batch_summarize(summary_chain, texts, {"max_concurrency": 1})

            print("Generating table summaries...")
            tables_html = [table.metadata.text_as_html for table in tables]
            table_summaries = safe_batch_summarize(summary_chain, tables_html, {"max_concurrency": 1})

            print("Generating image descriptions...")
            image_summaries = describe_images(image_chain, images)

            print("Setting up retriever...")
            self.retriever = setup_retriever(
                self.env_vars["GEMINI_API_KEY"],
                texts, text_summaries,
                tables, table_summaries,
                images, image_summaries
            )

            self.qa_chain = create_qa_chain(self.retriever, self.env_vars["GEMINI_API_KEY"])
            print("\nPDF processing complete! You can now ask questions.")

        except Exception as e:
            print(f"Error processing PDF: {e}")

    def _answer_question(self, question):
        """Answer a question using the processed PDF"""
        if not question:
            print("Please enter a question.")
            return

        print("Processing your question...")
        try:
            response = self.qa_chain.invoke(question)
            print("\nResponse:", response['response'])

            print("\n\nContext:")
            for text in response['context']['texts']:
                print(text.text)
                print("Page number: ", text.metadata.page_number)
                print("\n" + "-"*50 + "\n")

            for image in response['context']['images']:
                display_base64_image(image)

        except Exception as e:
            print(f"Error answering question: {e}")

    def ask_questions_loop(self):
        """Continuous question loop with mode selection"""
        self.interactive_learning()

    def interactive_learning(self):
        """Start interactive learning with multiple modes"""

        def chat_mode():
            print("\nChat Mode - Ask any questions about the document content")
            print("Type 'back' to return to mode selection\n")

            while True:
                question = input("Your question (or 'back' to exit): ").strip()
                if question.lower() == 'back':
                    break
                if question:
                    self._answer_question(question)

        def flashcard_mode():
            print("\nFlashcard Mode - Testing your knowledge of key concepts")
            print("Type 'back' to return to mode selection\n")
            prompt = (
                "Based on the document content, generate 10 important concept-definition pairs\n"
                "in the format: 'Concept: definition'. Be concise and focus on key information.\n"
                "Return ONLY the concept-definition pairs, one per line, with no additional text."
            )

            try:
                response = self.qa_chain.invoke(prompt)
                flashcards = [line.strip() for line in response['response'].split('\n') if ':' in line and line.strip()]
                if not flashcards:
                    print("Could not generate flashcards from this content.")
                    return

                current_card = 0
                show_definition = False

                while True:
                    if current_card >= len(flashcards):
                        print("\nYou've gone through all flashcards!")
                        break

                    card = flashcards[current_card]
                    parts = card.split(':', 1)

                    if len(parts) < 2:
                        current_card += 1
                        continue

                    if show_definition:
                        print(f"\nDefinition: {parts[1].strip()}")
                        print("\nOptions:")
                        print("1. Next card")
                        print("2. Previous card")
                        print("3. Back to menu")
                        choice = input("Your choice: ").strip()
                        if choice == '1':
                            current_card += 1
                            show_definition = False
                        elif choice == '2' and current_card > 0:
                            current_card -= 1
                            show_definition = False
                        elif choice == '3':
                            break
                        else:
                            print("Invalid choice")
                    else:
                        print(f"\nConcept: {parts[0].strip()}")
                        input("Press Enter to see definition...")
                        show_definition = True

            except Exception as e:
                print(f"Error generating flashcards: {e}")

        def quiz_mode():
            print("\nQuiz Mode - Test your knowledge with auto-generated questions")
            print("Type 'back' at any time to return to mode selection\n")

            prompt = (
                "Based on the document content, generate exactly 5 multiple choice questions\n"
                "that are strictly based on the main topics covered in the content.\n"
                "Format each question EXACTLY like this:\n\n"
                "Question 1: [question text]\n"
                "A) [option 1]\n"
                "B) [option 2]\n"
                "C) [option 3]\n"
                "D) [option 4]\n"
                "Correct: [letter of correct answer]\n"
                "Explanation: [brief explanation]"
            )

            try:
                response = self.qa_chain.invoke(prompt)
                quiz_content = response['response']

                questions = []
                current_question = None
                lines = [line.strip() for line in quiz_content.split('\n') if line.strip()]

                for line in lines:
                    if line.lower().startswith('question'):
                        if current_question:
                            questions.append(current_question)
                        current_question = {'text': line.split(':', 1)[1].strip(), 'options': [], 'correct': None, 'explanation': None}
                    elif line.upper().startswith(('A)', 'B)', 'C)', 'D)')):
                        current_question['options'].append(line)
                    elif line.lower().startswith('correct:'):
                        current_question['correct'] = line.split(':')[1].strip().upper()[0]
                    elif line.lower().startswith('explanation:'):
                        current_question['explanation'] = line.split(':', 1)[1].strip()

                if current_question:
                    questions.append(current_question)

                valid_questions = [
                    q for q in questions
                    if q['text'] and len(q['options']) == 4 and q['correct'] in ['A', 'B', 'C', 'D'] and q['explanation']
                ]

                if not valid_questions:
                    print("Could not generate proper quiz questions from this content.")
                    print("Here's what we got:")
                    print(quiz_content)
                    return

                score = 0
                for i, question in enumerate(valid_questions):
                    print(f"\nQuestion {i+1}: {question['text']}")
                    for option in question['options']:
                        print(option)

                    while True:
                        answer = input("\nYour answer (A/B/C/D) or 'back': ").upper().strip()
                        if answer == 'BACK':
                            return
                        if answer in ('A', 'B', 'C', 'D'):
                            break
                        print("Invalid choice. Please enter A, B, C, or D")

                    if answer == question['correct']:
                        print("\n✅ Correct!")
                        score += 1
                    else:
                        print(f"\n❌ Incorrect. The correct answer is {question['correct']}")
                        print(f"\nExplanation: {question['explanation']}")
                    input("\nPress Enter to continue...")

                print(f"\nQuiz complete! Your score: {score}/{len(valid_questions)}")

            except Exception as e:
                print(f"Error generating quiz: {e}")
                print("Full response was:")
                print(response['response'] if 'response' in response else "No response")

        # Main mode selection loop
        while True:
            print("\nSelect learning mode:")
            print("1. Chat Mode - Ask questions about the document")
            print("2. Flashcard Mode - Learn key concepts")
            print("3. Quiz Mode - Test your knowledge with questions")
            print("4. Exit")

            try:
                choice = input("Enter your choice (1-4): ").strip()
                if choice == '1':
                    chat_mode()
                elif choice == '2':
                    flashcard_mode()
                elif choice == '3':
                    quiz_mode()
                elif choice == '4':
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                
if __name__ == "__main__":
    processor = PDFProcessor()