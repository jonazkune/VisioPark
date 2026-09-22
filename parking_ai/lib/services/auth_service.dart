import '../models/app_user.dart';

class AuthService {
  static String detectorUrl = 'http://127.0.0.1:8000';
  static String? token;
  static AppUser? user;

  static bool get isLoggedIn => token != null && token!.isNotEmpty && user != null;

  static void saveSession({
    required String detector,
    required String sessionToken,
    required AppUser sessionUser,
  }) {
    detectorUrl = detector;
    token = sessionToken;
    user = sessionUser;
  }

  static void logout() {
    token = null;
    user = null;
  }
}
