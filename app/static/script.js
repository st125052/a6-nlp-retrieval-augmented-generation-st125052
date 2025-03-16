document.addEventListener("DOMContentLoaded", function() {
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatWindow = document.getElementById("chatWindow");
    const sendButton = document.getElementById("sendButton");
    const inputError = document.getElementById("inputError");
    
    chatInput.addEventListener("input", function() {
        sendButton.disabled = chatInput.value.trim().length === 0;
    });
    
    chatForm.addEventListener("submit", function(event) {
        event.preventDefault();
        const query = chatInput.value.trim();

        if (!query) {
            inputError.textContent = "Query cannot be empty.";
            return;
        }
        inputError.textContent = "";
        sendButton.disabled = true;
        chatInput.disabled = true;
        
        addChatMessage("You", query, "user-message");
        chatInput.value = "";
        
        const botMessageElement = addChatMessage("Bot", "Typing...", "bot-message");
        
        fetch(`/predict?search=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                const { answer, source_documents } = data;
                const fullMessage = `${answer}<br><br><strong>Source:</strong> ${source_documents}`;
                simulateTyping(botMessageElement, fullMessage, true);
            })
            .catch(error => {
                botMessageElement.innerHTML = `<strong>Bot:</strong> An error occurred. Please try again.`;
            })
            .finally(() => {
                sendButton.disabled = false;
                chatInput.disabled = false;
            });
    });
    
    function addChatMessage(sender, message, className) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("chat-message", className);
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return messageElement;
    }
    
    function simulateTyping(element, message, isHTML = false) {
        let i = 0;
        element.innerHTML = "<strong>Bot:</strong> ";
        function typeCharacter() {
            if (i < message.length) {
                if (isHTML) {
                    element.innerHTML = "<strong>Bot:</strong> " + message.substring(0, i + 1);
                } else {
                    element.innerHTML += message.charAt(i);
                }
                i++;
                setTimeout(typeCharacter, 50);
            }
        }
        typeCharacter();
    }
});