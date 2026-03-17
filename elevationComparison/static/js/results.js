// Additional JavaScript for results page interactivity

document.addEventListener('DOMContentLoaded', function() {
    // Add row highlighting on hover
    const table = document.getElementById('resultsTable');
    if (table) {
        const rows = table.querySelectorAll('tbody tr');
        rows.forEach(row => {
            row.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.01)';
                this.style.transition = 'transform 0.2s ease';
            });
            row.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });
        });
    }

    // Add tooltips for status badges
    const statusBadges = document.querySelectorAll('.status-badge');
    statusBadges.forEach(badge => {
        if (badge.classList.contains('warning')) {
            badge.title = 'This GCP point is outside the TIN coverage area';
        } else {
            badge.title = 'Elevation comparison successful';
        }
    });
});







