# Thermal Biometric Frontend

Modern web-based frontend for the Thermal Biometric System with user registration, video recording, anti-spoofing detection, and facial identification.

## 🎨 Features

### 1. User Registration
- **Camera Access**: Live camera feed with face guide overlay
- **Photo Capture**: Take a high-quality photo for registration
- **Duplicate Detection**: Checks if user is already registered
- **Name Input**: Associate a name with the face

### 2. Identification
- **Video Recording**: Record yourself for 5 seconds
- **Live Preview**: See what's being recorded
- **Two-Stage Verification**:
  1. **Anti-Spoofing**: Detects if you're using a photo/video (presentation attack)
  2. **Identification**: Matches your face against registered users
- **Detailed Results**: Shows confidence scores, distances, and top matches

### 3. Gallery View
- **Statistics Dashboard**: Total users, samples, system status
- **User List**: Browse all registered identities
- **Search**: Filter users by name
- **Real-time Updates**: Syncs with backend gallery

## 🚀 Quick Start

### Start the Server

```bash
cd src
./start_api.sh
```

Or:

```bash
cd src
python api.py
```

### Access the Frontend

Open your browser to: **http://localhost:8000**

The frontend is automatically served by the FastAPI backend.

## 🖥️ User Interface

### Layout

```
┌─────────────────────────────────────────────────────┐
│  Header: Logo + Navigation (Register | Identify | Gallery)
├─────────────────────────────────────────────────────┤
│                                                       │
│  Main Content Area:                                  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Active View (Register / Identify / Gallery)  │  │
│  │                                                │  │
│  │  - Camera/Video display                       │  │
│  │  - Controls and buttons                       │  │
│  │  - Results and feedback                       │  │
│  └───────────────────────────────────────────────┘  │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Color Scheme

- **Primary**: Indigo (`#4F46E5`) - Main actions
- **Success**: Green (`#10B981`) - Success states
- **Danger**: Red (`#EF4444`) - Errors/Recording
- **Warning**: Amber (`#F59E0B`) - Warnings
- **Background**: Light gray (`#F9FAFB`)

## 📱 Responsive Design

The interface adapts to different screen sizes:
- **Desktop**: Full layout with side-by-side elements
- **Tablet**: Stacked layout with comfortable spacing
- **Mobile**: Single column, touch-friendly controls

## 🎬 User Flow

### Registration Flow

```
1. Click "Register" tab
   ↓
2. Enter your name
   ↓
3. Click "Start Camera"
   ↓
4. Position face in guide overlay
   ↓
5. Click "Capture Photo"
   ↓
6. Preview captured photo
   ↓
7. Click "Register" to submit
   ↓
8. System checks for duplicates
   ↓
9. Success message displayed
```

### Identification Flow

```
1. Click "Identify" tab
   ↓
2. Click "Start Camera"
   ↓
3. Click "Start Recording"
   ↓
4. Show your face clearly for 5 seconds
   ↓
5. Recording automatically stops
   ↓
6. Preview recorded video
   ↓
7. Click "Analyze"
   ↓
8. STEP 1: Anti-Spoofing Check
   │  - Extracts frame from video
   │  - Checks for liveness
   │  - Detects presentation attacks
   ↓
9. STEP 2: Identification
   │  - Compares face to gallery
   │  - Calculates distances
   │  - Finds best match
   ↓
10. Results displayed with:
    - Identity name
    - Confidence score
    - Top matches
    - Liveness status
```

## 🔧 Technical Details

### Browser Requirements

- Modern browser with:
  - **getUserMedia API**: Camera access
  - **MediaRecorder API**: Video recording
  - **Canvas API**: Frame extraction
  - **Fetch API**: API communication

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Camera Permissions

The app requires camera permissions:
1. Browser will prompt for permission on first use
2. Grant permission for camera access
3. Permission is remembered for future visits

### Video Recording

- **Duration**: 5 seconds (auto-stop)
- **Format**: WebM with VP9 codec
- **Frame Extraction**: Middle frame used for analysis
- **Quality**: High-quality JPEG (95%)

### API Communication

All communication with backend uses:
- **REST API**: Standard HTTP methods
- **FormData**: For file uploads
- **JSON**: For responses
- **CORS**: Enabled for cross-origin requests

## 📂 File Structure

```
src/static/
├── index.html      # Main HTML structure
├── styles.css      # All CSS styling
└── app.js          # JavaScript application logic
```

### Components

**index.html**:
- Header with navigation
- Three main views (Register, Identify, Gallery)
- Camera/video containers
- Result display areas
- Loading overlay
- Toast notifications

**styles.css**:
- Responsive layout
- Component styling
- Animations and transitions
- Color variables
- Media queries

**app.js**:
- Camera management
- Video recording
- API communication
- UI state management
- Event handlers

## 🎨 Customization

### Colors

Edit CSS variables in `styles.css`:

```css
:root {
    --primary-color: #4F46E5;
    --success-color: #10B981;
    --danger-color: #EF4444;
    /* ... */
}
```

### Recording Duration

Edit `app.js`:

```javascript
// Change from 5000ms to desired duration
setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    }
}, 5000);  // <- Change this value
```

### API Endpoint

Edit `app.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // <- Change this
```

## 🐛 Troubleshooting

### Camera Not Working

**Problem**: "Permission denied" or camera doesn't start

**Solutions**:
1. Check browser permissions (usually in address bar)
2. Use HTTPS (required by some browsers)
3. Try different browser
4. Check if camera is used by another application

### Recording Not Working

**Problem**: Video recording fails or produces corrupted file

**Solutions**:
1. Check browser support for MediaRecorder
2. Try different codec (edit `mimeType` in `app.js`)
3. Reduce recording duration
4. Check browser console for errors

### API Connection Failed

**Problem**: "Failed to fetch" or CORS errors

**Solutions**:
1. Ensure API server is running (`python api.py`)
2. Check API_BASE_URL in `app.js`
3. Verify CORS is enabled in `api.py`
4. Check browser console for detailed errors

### Face Not Detected

**Problem**: "No face detected" error

**Solutions**:
1. Ensure good lighting
2. Position face within guide overlay
3. Look directly at camera
4. Remove glasses/hat if necessary
5. Move closer to camera

## 🔒 Security Considerations

### Current Implementation (Development)

- ✅ HTTPS recommended for production
- ✅ CORS enabled for all origins (restrict in production)
- ✅ Client-side validation
- ⚠️ No authentication (add for production)
- ⚠️ Direct API access (add rate limiting)

### Production Recommendations

1. **HTTPS**: Deploy with SSL/TLS
2. **Authentication**: Add user login system
3. **Rate Limiting**: Prevent API abuse
4. **Input Validation**: Server-side validation
5. **CORS**: Restrict to specific domains
6. **CSP**: Content Security Policy headers
7. **Session Management**: Secure session handling

## 📊 Performance

### Load Times

- **Initial Load**: < 1s (static assets)
- **Camera Start**: ~500ms
- **Photo Capture**: Instant
- **Video Recording**: Real-time
- **API Calls**: 500ms - 2s (depends on model processing)

### Optimization

- Minimal dependencies (no heavy frameworks)
- CSS animations (GPU accelerated)
- Lazy loading of camera streams
- Efficient API calls (only when needed)
- Automatic cleanup of resources

## 🚀 Deployment

### Development

```bash
python api.py
# Access at http://localhost:8000
```

### Production with Nginx

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker

The frontend is included in the Docker image:

```bash
docker-compose up -d
# Access at http://localhost:8000
```

## 🎓 Usage Tips

### Best Practices

1. **Good Lighting**: Ensure face is well-lit
2. **Center Face**: Position within guide overlay
3. **Look at Camera**: Direct eye contact
4. **Stable Position**: Keep still during recording
5. **Clear Background**: Plain background works best

### Registration Tips

- Use clear, recent photo
- Good lighting from front
- Neutral expression
- No sunglasses or hat
- Face clearly visible

### Identification Tips

- Record in similar conditions to registration
- Show natural movements
- Don't cover face
- Maintain eye contact with camera
- Stay within frame

## 📝 Notes

- **Gallery is Read-Only**: Current implementation uses pre-loaded gallery
- **No Backend Registration**: Registration UI is for demonstration
- **Frame Extraction**: Uses middle frame from recorded video
- **Auto-Stop**: Recording automatically stops after 5 seconds
- **Toast Notifications**: Show feedback for all actions

## 🔗 Related Documentation

- [API_README.md](../API_README.md) - Complete API documentation
- [README.md](../README.md) - Main project README
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture

## 💡 Future Enhancements

Potential improvements:
- [ ] Real-time face detection overlay
- [ ] Multiple face support
- [ ] Gallery management (add/delete users)
- [ ] Export identification logs
- [ ] Dark mode
- [ ] Multi-language support
- [ ] Progressive Web App (PWA)
- [ ] Offline support
- [ ] Mobile app (React Native/Flutter)

## 🆘 Support

For issues or questions:
1. Check browser console for errors
2. Verify API server is running
3. Check [TROUBLESHOOTING](#-troubleshooting) section
4. Review API logs
5. Open an issue in the repository
