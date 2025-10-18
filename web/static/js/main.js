// VMEvalKit Dashboard - Interactive Features

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeVideoPlayers();
    initializeFilters();
    initializeSearch();
    initializeLazyLoading();
});

// Video player enhancements
function initializeVideoPlayers() {
    const videos = document.querySelectorAll('video');
    
    videos.forEach(video => {
        // Add play/pause on click
        video.addEventListener('click', function() {
            if (this.paused) {
                this.play();
            } else {
                this.pause();
            }
        });
        
        // Add loading indicator
        video.addEventListener('loadstart', function() {
            this.classList.add('loading');
        });
        
        video.addEventListener('loadeddata', function() {
            this.classList.remove('loading');
        });
        
        // Handle errors
        video.addEventListener('error', function() {
            console.error('Video failed to load:', this.src);
            const container = this.closest('.result-video') || this.closest('.comparison-section');
            if (container) {
                container.innerHTML = '<div class="video-error">❌ Video failed to load</div>';
            }
        });
    });
}

// Filter functionality for tables
function initializeFilters() {
    const filterInputs = document.querySelectorAll('[data-filter]');
    
    filterInputs.forEach(input => {
        input.addEventListener('input', function() {
            const filterValue = this.value.toLowerCase();
            const targetTable = document.querySelector(this.dataset.filter);
            
            if (targetTable) {
                const rows = targetTable.querySelectorAll('tbody tr');
                
                rows.forEach(row => {
                    const text = row.textContent.toLowerCase();
                    row.style.display = text.includes(filterValue) ? '' : 'none';
                });
            }
        });
    });
}

// Search functionality
function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', debounce(function() {
        const query = this.value.toLowerCase();
        const cards = document.querySelectorAll('.result-card, .domain-card, .comparison-card');
        
        cards.forEach(card => {
            const text = card.textContent.toLowerCase();
            card.style.display = text.includes(query) ? '' : 'none';
        });
    }, 300));
}

// Lazy loading for images and videos
function initializeLazyLoading() {
    if ('IntersectionObserver' in window) {
        const lazyElements = document.querySelectorAll('video[preload="metadata"]');
        
        const lazyLoadObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const video = entry.target;
                    video.preload = 'auto';
                    lazyLoadObserver.unobserve(video);
                }
            });
        });
        
        lazyElements.forEach(el => lazyLoadObserver.observe(el));
    }
}

// Utility: Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func.apply(this, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showNotification('Failed to copy', 'error');
    });
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#2563eb'};
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Sort table columns
function sortTable(table, column, ascending = true) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort((a, b) => {
        const aText = a.cells[column].textContent.trim();
        const bText = b.cells[column].textContent.trim();
        
        // Try to parse as numbers
        const aNum = parseFloat(aText);
        const bNum = parseFloat(bText);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return ascending ? aNum - bNum : bNum - aNum;
        }
        
        // String comparison
        return ascending ? 
            aText.localeCompare(bText) : 
            bText.localeCompare(aText);
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K for search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('search-input');
        if (searchInput) searchInput.focus();
    }
    
    // Escape to clear search
    if (e.key === 'Escape') {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.value = '';
            searchInput.dispatchEvent(new Event('input'));
        }
    }
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
    
    .video-error {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 200px;
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        font-weight: 600;
    }
    
    video.loading {
        opacity: 0.5;
    }
`;
document.head.appendChild(style);

// Export functions for use in templates
window.VMEvalKit = {
    copyToClipboard,
    showNotification,
    sortTable
};

