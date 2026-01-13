// Auto-detecting environment configuration
// Works for both local Docker and Render deployment
export const environment = {
    production: true,
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
