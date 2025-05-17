// Highlight active link in navigation
document.addEventListener('DOMContentLoaded', function() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
    });
    
    // For range sliders in the form
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        const valueDisplay = document.getElementById(`${input.id}-value`);
        if (valueDisplay) {
            input.addEventListener('input', () => {
                valueDisplay.textContent = input.value;
            });
        }
    });
    
    // Prevent form submission if no content in journal
    const journalForm = document.querySelector('form[action="/journal"]');
    if (journalForm) {
        journalForm.addEventListener('submit', function(e) {
            const content = document.getElementById('content').value.trim();
            if (content.length < 10) {
                e.preventDefault();
                alert('Пожалуйста, напишите не менее 10 символов в дневнике.');
            }
        });
    }
}); 

