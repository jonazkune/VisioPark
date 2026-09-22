import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return android;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.windows:
        return android;
      default:
        return android;
    }
  }

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyC8FOKuyenRYW5Xtxse75Y4Mw8nCB8SfII',
    appId: '1:513791791890:android:9484fce010793b8d3b99cb',
    messagingSenderId: '513791791890',
    projectId: 'parkingia-b8bf6',
    storageBucket: 'parkingia-b8bf6.firebasestorage.app',
  );
}
