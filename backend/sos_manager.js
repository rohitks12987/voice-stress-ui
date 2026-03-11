const SOS_API_URL = '/api/user/sos';
const CONTACTS_API_URL = '/api/user/contacts';

// Helper to get current logged in user email
function getUserEmail() {
    const profile = localStorage.getItem('userProfile');
    if (profile) {
        return JSON.parse(profile).email;
    }
    return localStorage.getItem('user_email');
}

async function activateSOS() {
    const email = getUserEmail();
    if (!email) {
        alert("Error: User not logged in. Cannot send SOS.");
        return;
    }

    // Find the button to update UI state
    const sosBtn = document.querySelector('.sos-btn') || document.getElementById('sos-btn');
    const originalText = sosBtn ? sosBtn.innerText : "SOS";
    
    if(sosBtn) {
        sosBtn.innerText = "📍 LOCATING...";
        sosBtn.disabled = true;
    }

    // 1. Get Location
    let locationStr = "Unknown Location";
    if (navigator.geolocation) {
        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    timeout: 10000,
                    maximumAge: 0,
                    enableHighAccuracy: true
                });
            });
            locationStr = `${position.coords.latitude}, ${position.coords.longitude}`;
            // Optional: You can integrate Google Maps Link here if desired
            locationStr = `https://www.google.com/maps?q=${position.coords.latitude},${position.coords.longitude}`;
        } catch (error) {
            console.warn("GPS Location failed:", error);
            locationStr = "GPS Access Denied";
        }
    }

    if(sosBtn) sosBtn.innerText = "📡 SENDING...";

    // 2. Send Alert to Backend
    try {
        const response = await fetch(SOS_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_email: email,
                location: locationStr
            })
        });

        const result = await response.json();

        if (response.ok) {
            alert(`🚨 SOS SENT SUCCESSFULLY!\n\n${result.message}`);
        } else {
            alert(`⚠️ Failed to send SOS: ${result.message}`);
        }
    } catch (error) {
        console.error("SOS System Error:", error);
        alert("❌ Network Error: Could not reach Emergency Server.");
    } finally {
        if(sosBtn) {
            sosBtn.innerText = originalText;
            sosBtn.disabled = false;
        }
    }
}

// Attach listener if button exists immediately
document.addEventListener("DOMContentLoaded", () => {
    const sosBtn = document.querySelector('.sos-btn') || document.getElementById('sos-btn');
    if (sosBtn) {
        sosBtn.onclick = activateSOS;
    }
});