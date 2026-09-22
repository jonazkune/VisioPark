import 'package:cloud_firestore/cloud_firestore.dart';

class FirebaseService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  // Bilduma batetik datu guztiak lortu (Stream moduan)
  Stream<QuerySnapshot> getCollectionStream(String collectionPath) {
    return _firestore.collection(collectionPath).snapshots();
  }

  Stream<DocumentSnapshot<Map<String, dynamic>>> getDocumentStream(String collectionPath, String docId) {
    return _firestore.collection(collectionPath).doc(docId).snapshots();
  }

  // Dokumentu berri bat gehitu bilduma batera
  Future<DocumentReference> addDocument(
    String collectionPath,
    Map<String, dynamic> data,
  ) {
    return _firestore.collection(collectionPath).add(data);
  }

  Future<void> setDocument(
    String collectionPath,
    String docId,
    Map<String, dynamic> data,
  ) {
    return _firestore.collection(collectionPath).doc(docId).set(data, SetOptions(merge: true));
  }

  Future<void> updateDocument(
    String collectionPath,
    String docId,
    Map<String, dynamic> data,
  ) {
    return _firestore.collection(collectionPath).doc(docId).update(data);
  }
}
