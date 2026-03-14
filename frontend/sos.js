/**
 * SOS Emergency System
 * Handles geolocation and communication with the backend emergency API.
 */

const SOS_CONFIG = {
    API_URL: "/api",
    ENDPOINTS: {
        SOS: "/user/sos",
        CONTACTS: "/user/contacts"
    }
};

async function triggerSOS() {
    const sosBtn = document.querySelector('.sos-btn');
    if (!sosBtn || sosBtn.disabled) return;

    // 1. User Confirmation
    if (!confirm("🚨 SEND EMERGENCY ALERT?\n\nThis will send your current location to your registered emergency contacts.")) {
        return;
    }

    // 2. UI Feedback - Lock Button
    const originalText = sosBtn.innerText;
    sosBtn.innerText = "📍 LOCATING...";
    sosBtn.disabled = true;

    try {
        // 3. Identify User
        const userProfile = JSON.parse(localStorage.getItem('userProfile')) || {};
        const userEmail = userProfile.email || localStorage.getItem('user_email');

        if (!userEmail) {
            throw new Error("User email not found. Please update your profile in Settings.");
        }

        // 4. Get Geolocation
        const position = await new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error("Geolocation is not supported by your browser."));
            } else {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 10000, // 10 seconds timeout
                    maximumAge: 0
                });
            }
        });

        const { latitude, longitude } = position.coords;
        const locationMapLink = `https://www.google.com/maps?q=${latitude},${longitude}`;

        // 5. Send Alert to Backend
        sosBtn.innerText = "📡 SENDING...";
        
        const response = await fetch(`${SOS_CONFIG.API_URL}${SOS_CONFIG.ENDPOINTS.SOS}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_email: userEmail,
                location: locationMapLink,
                coordinates: { lat: latitude, lng: longitude }
            })
        });

        const result = await response.json();

        if (response.ok) {
            alert(`✅ EMERGENCY ALERT SENT!\n\n${result.message || "Help is on the way. Contacts notified."}`);
        } else {
            throw new Error(result.message || "Server rejected the request.");
        }

    } catch (error) {
        console.error("SOS Error:", error);
        alert(`❌ ALERT FAILED: ${error.message}\n\nPlease contact emergency services directly.`);
    } finally {
        // 6. Reset UI
        sosBtn.innerText = originalText;
        sosBtn.disabled = false;
    }
}

// Initialize Listener
document.addEventListener('DOMContentLoaded', () => {
    const sosBtn = document.querySelector('.sos-btn');
    if (sosBtn) {
        sosBtn.addEventListener('click', triggerSOS);
    }
});