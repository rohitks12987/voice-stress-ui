// API URLs - use relative paths since same origin
const BACKEND_URL = window.location.origin || '';
const SOS_API_URL = `${BACKEND_URL}/api/user/sos`;
const CONTACTS_API_URL = `${BACKEND_URL}/api/user/contacts`;

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

// Initialize SOS system
function initSOSSystem() {
    const sosBtn = document.querySelector('.btn-sos') || document.getElementById('sos-btn');
    const addContactBtn = document.getElementById('add-contact-btn');
    
    if (sosBtn) {
        sosBtn.onclick = activateSOS;
    }
    if (addContactBtn) {
        addContactBtn.onclick = promptAddEmergencyContact;
    }
}

// Start initialization
document.addEventListener("DOMContentLoaded", initSOSSystem);

// NEW FEATURE: Allow Patient to add an Emergency Contact (Telegram, Email, or Phone)
async function promptAddEmergencyContact() {
    const email = getUserEmail();
    console.log("User email detected:", email);
    if (!email) {
        alert("Please log in first to add a contact. Debug: " + email);
        return;
    }

    const name = prompt("Enter Emergency Contact Name (e.g., Brother, Doctor):");
    if (!name) return;
    
    const contactMethod = prompt("Enter contact method:\n'T' for Telegram\n'E' for Email\n'P' for Phone");
    if (!contactMethod) return;
    
    let contactValue;
    if (contactMethod.toUpperCase() === 'T') {
        contactValue = prompt("Enter Telegram Chat ID (e.g., 123456789):");
    } else if (contactMethod.toUpperCase() === 'P') {
        contactValue = prompt("Enter Phone Number with country code (e.g., +919876543210):");
    } else {
        contactValue = prompt("Enter Emergency Contact Email (e.g., john@gmail.com):");
    }
    if (!contactValue) return;
    
    const payload = {
        user_email: email,
        name: name,
        relationship: "Emergency"
    };
    
    if (contactMethod.toUpperCase() === 'T') {
        payload.telegram_chat_id = contactValue;
    } else if (contactMethod.toUpperCase() === 'P') {
        payload.phone = contactValue;
    } else {
        payload.email = contactValue;
    }
    
    console.log("Sending:", payload);
    
    const response = await fetch(CONTACTS_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const result = await response.json();
    console.log("Response:", result);
    alert(response.ok ? "✅ Emergency Contact Saved Successfully!" : "⚠️ Error: " + result.message);
}

// View contacts
async function viewEmergencyContacts() {
    const email = getUserEmail();
    if (!email) {
        alert("Please log in first.");
        return;
    }
    
    try {
        const response = await fetch(`${CONTACTS_API_URL}?user_email=${encodeURIComponent(email)}`);
        const data = await response.json();
        
        if (data.contacts && data.contacts.length > 0) {
            let msg = "Your Emergency Contacts:\n\n";
            data.contacts.forEach((c, i) => {
                let contactInfo = "";
                if (c.email) contactInfo = `📧 ${c.email}`;
                else if (c.telegram_chat_id) contactInfo = `📱 Telegram: ${c.telegram_chat_id}`;
                else if (c.phone) contactInfo = `📞 ${c.phone}`;
                msg += `${i+1}. ${c.name} - ${contactInfo}\n`;
            });
            alert(msg);
        } else {
            alert("No emergency contacts added yet.");
        }
    } catch (e) {
        alert("Error fetching contacts: " + e.message);
    }
}

// Initialize
function initSOSSystem() {
    const sosBtn = document.querySelector('.btn-sos') || document.getElementById('sos-btn');
    const addContactBtn = document.getElementById('add-contact-btn');
    const viewContactBtn = document.getElementById('view-contacts-btn');
    
    if (sosBtn) {
        sosBtn.onclick = activateSOS;
    }
    if (addContactBtn) {
        addContactBtn.onclick = promptAddEmergencyContact;
    }
    if (viewContactBtn) {
        viewContactBtn.onclick = viewEmergencyContacts;
    }
}

// Make it globally available so you can use it anywhere in your UI
window.promptAddEmergencyContact = promptAddEmergencyContact;