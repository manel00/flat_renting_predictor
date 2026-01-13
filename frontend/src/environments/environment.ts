// Default environment with auto-detection
export const environment = {
    production: false,
    // Auto-detect backend URL based on hostname
    get apiUrl(): string {
        // If running on Render (hostname contains 'onrender.com')
        if (typeof window !== 'undefined' && window.location.hostname.includes('onrender.com')) {
            return 'https://housing-backend-fu9f.onrender.com/api';
        }
        // If running locally (Docker or localhost)
        return '/api';
    }
};
