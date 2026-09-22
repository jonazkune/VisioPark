class AppUser {
  final String username;
  final String name;
  final String role;
  final List<String> parkingIds;

  AppUser({
    required this.username,
    required this.name,
    required this.role,
    this.parkingIds = const [],
  });

  bool get isAdmin => role == 'admin';
  bool get isOwner => role == 'owner' || isAdmin;
  bool get canCreateParking => isAdmin || role == 'owner';

  factory AppUser.fromJson(Map<String, dynamic> json) {
    return AppUser(
      username: json['username']?.toString() ?? '',
      name: json['name']?.toString() ?? json['username']?.toString() ?? '',
      role: json['role']?.toString() ?? 'user',
      parkingIds: [
        for (final item in (json['parking_ids'] as List? ?? [])) item.toString(),
      ],
    );
  }
}
