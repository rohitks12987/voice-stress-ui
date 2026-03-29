const SOS_API_URL = '/api/user/sos';
const CONTACTS_API_URL = '/api/user/contacts';

// Helper to get current logged in user email
function getUserEmail() {
    const profile = localStorage.getItem('userProfile');
    if (profile) {
        return JSON.parse(profile).email;
    }
    const userEmail = localStorage.getItem('user_email');
    if (userEmail) return userEmail;
    return sessionStorage.getItem('activeUser');
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

let initRetries = 0;

// Initialize SOS and inject the Add Contact button reliably
function initSOSSystem() {
    const sosBtn = document.querySelector('.sos-btn') || document.getElementById('sos-btn');
    
    if (sosBtn) {
        sosBtn.onclick = activateSOS;

        // AUTO-INJECT: Automatically add the "Add Contact" button to the dashboard
        if (!document.getElementById('add-contact-btn')) {
            const addBtn = document.createElement('button');
            addBtn.id = 'add-contact-btn';
            addBtn.innerHTML = '➕ Add Emergency Contact';
            addBtn.style.cssText = 'margin-top: 10px; padding: 10px; background-color: #0284c7; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.2); display: block; text-align: center;';
            addBtn.onclick = promptAddEmergencyContact;
            
            // Place it right below the SOS button
            if (sosBtn.parentNode) {
                sosBtn.parentNode.insertBefore(addBtn, sosBtn.nextSibling);
            }
        }
    } else if (initRetries < 20) {
        // If the dashboard is loading dynamically, wait 500ms and try again
        initRetries++;
        setTimeout(initSOSSystem, 500);
    }

    // Attach listener if you already have a button with id="add-contact-btn" hardcoded
    const addContactBtn = document.getElementById('add-contact-btn');
    if (addContactBtn && !addContactBtn.onclick) {
        addContactBtn.onclick = promptAddEmergencyContact;
    }
}

// Start initialization
document.addEventListener("DOMContentLoaded", initSOSSystem);
initSOSSystem();

// NEW FEATURE: Allow Patient to add an Emergency Email Contact
async function promptAddEmergencyContact() {
    const email = getUserEmail();
    console.log("User email detected:", email);
    if (!email) {
        alert("Please log in first to add a contact. Debug: " + email);
        return;
    }

    const name = prompt("Enter Emergency Contact Name (e.g., Brother, Doctor):");
    if (!name) return;
    
    const contactEmail = prompt("Enter Emergency Contact Email (e.g., john@gmail.com):");
    if (!contactEmail) return;
    
    console.log("Sending:", { user_email: email, name: name, email: contactEmail });
    
    const response = await fetch(CONTACTS_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_email: email, name: name, email: contactEmail, relationship: "Emergency" })
    });
    const result = await response.json();
    console.log("Response:", result);
    alert(response.ok ? "✅ Emergency Contact Saved Successfully!" : "⚠️ Error: " + result.message);
}

// Make it globally available so you can use it anywhere in your UI
window.promptAddEmergencyContact = promptAddEmergencyContact;